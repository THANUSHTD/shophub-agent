"""R2: Orders sub-agent exposed over Google's A2A protocol (JSON-RPC 2.0 over HTTP).

Deterministic: contains NO LLM. The executor maps a skill request to a Store API
call, forwards the bearer token from the incoming HTTP Authorization header,
and returns every result wrapped in a DataPart envelope.

Serves the Agent Card at /.well-known/agent-card.json. Port 8001.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import FastAPI
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct, Value

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.types.a2a_pb2 import (
    AgentCard,
    AgentCapabilities,
    AgentInterface,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Role,
    SendMessageRequest,
    SendMessageConfiguration,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

from config import MERCHANT_TOKENS
import config as _config
from guardrails import sanitize_order

STORE_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def _now() -> datetime:
    return datetime.now(timezone.utc)


SKILLS = [
    {
        "id": "list_orders",
        "name": "List orders",
        "description": "List the merchant's orders, optionally filtered by status or age.",
        "tags": ["orders", "list", "read"],
    },
    {
        "id": "lookup_order",
        "name": "Lookup order",
        "description": "Look up a single order by id.",
        "tags": ["orders", "lookup", "read"],
    },
    {
        "id": "search_customer_orders",
        "name": "Search customer orders",
        "description": "Find orders by customer name (case-insensitive substring), newest first.",
        "tags": ["orders", "search", "read"],
    },
    {
        "id": "refund_order",
        "name": "Refund order",
        "description": "Propose/refund an order (executed only after human approval).",
        "tags": ["orders", "refund", "write"],
    },
    {
        "id": "cancel_order",
        "name": "Cancel order",
        "description": "Propose/cancel an order (executed only after human approval).",
        "tags": ["orders", "cancel", "write"],
    },
]


def _envelope(ok: bool, data: Any = None, error: str | None = None, message: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": ok}
    if data is not None:
        result["data"] = data
    if error is not None:
        result["error"] = error
    if message is not None:
        result["message"] = message
    return result


def _part_from_envelope(envelope: dict[str, Any]) -> Part:
    sv = Struct()
    sv.update(envelope)
    return Part(data=Value(struct_value=sv))


def _artifact(envelope: dict[str, Any]) -> Artifact:
    return Artifact(parts=[_part_from_envelope(envelope)], name="result")


def _completed_task(context: RequestContext, envelope: dict[str, Any]) -> Task:
    task_id = context.task_id or str(uuid.uuid4())
    context_id = context.context_id or str(uuid.uuid4())
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        artifacts=[_artifact(envelope)],
    )


def _bearer_token(context: RequestContext) -> str | None:
    """Read the bearer token from the incoming HTTP Authorization header.

    The SDK populates ServerCallContext.state['headers'] with the request
    headers. Returns the raw token with any "Bearer " prefix stripped, so it
    can be re-prefixed when forwarded to the Store API.
    """
    headers = context.call_context.state.get("headers") or {}
    auth = headers.get("authorization") or headers.get("Authorization")
    if auth is not None and auth.lower().startswith("bearer "):
        auth = auth[len("bearer "):]
    return auth


async def _store_request(client: httpx.AsyncClient, method: str, path: str, token: str | None, **kwargs):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return await client.request(method, f"{os.environ.get('STORE_API_URL', 'http://store-api:8002')}{path}", headers=headers, **kwargs)


class OrdersExecutor(AgentExecutor):
    def __init__(self):
        self._store = httpx.AsyncClient(timeout=STORE_TIMEOUT)

    async def _me(self, client: httpx.AsyncClient, token: str | None) -> dict[str, Any] | None:
        try:
            resp = await _store_request(client, "GET", "/auth/me", token)
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        return resp.json()

    async def _dispatch(self, client: httpx.AsyncClient, token: str | None, skill: str, args: dict[str, Any]) -> dict[str, Any]:
        me = await self._me(client, token)
        if me is None or me.get("merchant") is None:
            return _envelope(False, error="NO_MERCHANT", message="No merchant scoped to bearer token")
        merchant = me["merchant"]

        try:
            if skill == "list_orders":
                return await self._list_orders(client, token, merchant, args)
            if skill == "lookup_order":
                return await self._lookup_order(client, token, merchant, args)
            if skill == "search_customer_orders":
                return await self._search_customer_orders(client, token, merchant, args)
            if skill == "refund_order":
                return await self._write(client, token, merchant, "/orders/refund", args)
            if skill == "cancel_order":
                return await self._write(client, token, merchant, "/orders/cancel", args)
            return _envelope(False, error="INVALID_STATE", message=f"Unknown skill {skill}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return _envelope(False, error="ORDER_NOT_FOUND", message="Order not found")
            if e.response.status_code == 409:
                return _envelope(False, error="INVALID_STATE", message="Invalid state transition")
            return _envelope(False, error="UPSTREAM_ERROR", message=str(e))
        except Exception as e:  # pragma: no cover - defensive
            return _envelope(False, error="UPSTREAM_ERROR", message=str(e))

    async def _list_orders(self, client, token, merchant, args):
        params = {}
        if args.get("status"):
            params["status"] = args["status"]
        resp = await _store_request(client, "GET", "/orders", token, params=params)
        resp.raise_for_status()
        orders = resp.json()
        older_than = args.get("older_than_days")
        if older_than is not None:
            cutoff = _now() - timedelta(days=float(older_than))
            filtered = []
            for o in orders:
                pa = o.get("placed_at")
                if pa is None:
                    continue
                try:
                    dt = datetime.fromisoformat(pa.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if dt <= cutoff:
                    filtered.append(o)
            orders = filtered
        return _envelope(True, data=[sanitize_order(o) for o in orders])

    async def _lookup_order(self, client, token, merchant, args):
        oid = args.get("id")
        resp = await _store_request(client, "GET", "/orders/lookup", token, params={"id": oid})
        resp.raise_for_status()
        body = resp.json()
        if not body.get("status"):
            return _envelope(False, error="ORDER_NOT_FOUND", message="Order not found")
        return _envelope(True, data=sanitize_order(body["data"]))

    async def _search_customer_orders(self, client, token, merchant, args):
        name = args.get("name", "")
        resp = await _store_request(client, "GET", "/orders", token, params={"status": None})
        resp.raise_for_status()
        orders = resp.json()
        name_l = name.lower()
        matched = [o for o in orders if o["merchant"] == merchant and name_l in (o.get("customer_name") or "").lower()]
        matched.sort(key=lambda o: o.get("placed_at") or "", reverse=True)
        return _envelope(True, data=[sanitize_order(o) for o in matched])

    async def _write(self, client, token, merchant, path, args):
        oid = args.get("id")
        resp = await _store_request(client, "POST", path, token, json={"id": oid})
        if resp.status_code == 404:
            return _envelope(False, error="ORDER_NOT_FOUND", message="Order not found")
        if resp.status_code == 409:
            return _envelope(False, error="INVALID_STATE", message="Invalid state transition")
        resp.raise_for_status()
        return _envelope(True, data=sanitize_order(resp.json()))

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        token = _bearer_token(context)
        # Parse the skill request from the incoming DataPart.
        skill = None
        args: dict[str, Any] = {}
        parts = context.message.parts if context.message and context.message.parts else []
        for part in parts:
            if part.data is not None:
                try:
                    payload = json_format.MessageToDict(part.data)
                except Exception:
                    payload = {}
                skill = payload.get("skill")
                args = payload.get("args") or {}
                break

        envelope: dict[str, Any]
        if skill is None:
            envelope = _envelope(False, error="INVALID_STATE", message="No skill specified in DataPart")
        else:
            # One shared httpx client per call.
            async with httpx.AsyncClient(timeout=STORE_TIMEOUT) as client:
                envelope = await self._dispatch(client, token, skill, args)

        await event_queue.enqueue_event(_completed_task(context, envelope))

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        await event_queue.enqueue_event(
            Task(
                id=context.task_id or str(uuid.uuid4()),
                context_id=context.context_id or str(uuid.uuid4()),
                status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
            )
        )


def create_app() -> FastAPI:
    import os
    public_url = os.environ.get("ORDERS_AGENT_PUBLIC_URL") or "http://orders-agent:8001"
    card = AgentCard(
        name="ShopHub Orders Agent",
        description="Read-only and proposal-only order operations for a single merchant tenant over A2A.",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        security_schemes={
            "bearer": {"http_auth_security_scheme": {"scheme": "bearer", "bearer_format": "JWT"}}
        },
        security_requirements=[{"schemes": {"bearer": {"list": []}}}],
        supported_interfaces=[
            AgentInterface(url=public_url, protocol_binding="JSONRPC", protocol_version="1.0")
        ],
        default_input_modes=["application/a2a+json"],
        default_output_modes=["application/a2a+json"],
        skills=[
            AgentSkill(
                id=s["id"],
                name=s["name"],
                description=s["description"],
                tags=s["tags"],
            )
            for s in SKILLS
        ],
    )
    app = FastAPI(title="ShopHub Orders Agent (A2A)")
    executor = OrdersExecutor()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"),
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("ORDERS_AGENT_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
