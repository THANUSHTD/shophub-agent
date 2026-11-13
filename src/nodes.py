from __future__ import annotations
import contextvars
import json
import os
import uuid
from typing import Any, Annotated, TypedDict
import httpx
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from guardrails import (
    redact_pii,
    require_merchant,
    sanitize_order,
    untrusted_wrap,
)
from prompts import SYSTEM_PROMPT


_token_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("shophub_token", default=None)


def set_token(token: str | None) -> None:
    _token_var.set(token)


def get_token() -> str | None:
    return _token_var.get()


class State(TypedDict):
    messages: Annotated[list[Any], add_messages]
    pending_proposal: dict[str, Any] | None


class OrdersA2AClient:
    def __init__(self):
        self._card: Any = None
        self._card_url: str | None = None

    async def _ensure_card(self, client: httpx.AsyncClient):
        if self._card is not None:
            return
        from a2a.client.card_resolver import A2ACardResolver
        resolver = A2ACardResolver(client, os.environ.get("ORDERS_AGENT_URL", "http://orders-agent:8001"))
        self._card = await resolver.get_agent_card()
        ifaces = list(getattr(self._card, "supported_interfaces", []) or [])
        for iface in ifaces:
            if getattr(iface, "protocol_binding", "") == "JSONRPC":
                self._card_url = iface.url or f"{os.environ.get('ORDERS_AGENT_URL', 'http://orders-agent:8001').rstrip('/')}/"
                break
        if self._card_url is None:
            self._card_url = f"{os.environ.get('ORDERS_AGENT_URL', 'http://orders-agent:8001').rstrip('/')}/"

    async def call_skill(
        self,
        skill: str,
        args: dict[str, Any],
        token: str | None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0)) as client:
                await self._ensure_card(client)
                from a2a.client import create_client
                from a2a.client.client import ClientConfig, ClientCallContext
                from a2a.types.a2a_pb2 import (
                    Message, Part, Role, SendMessageRequest,
                    SendMessageConfiguration,
                )
                from google.protobuf.struct_pb2 import Struct

                client_obj = await create_client(
                    self._card,
                    client_config=ClientConfig(streaming=False, polling=False),
                )
                try:
                    sv = Struct()
                    sv.update({"skill": skill, "args": args})
                    part = Part(data=Value(struct_value=sv))
                    msg = Message(message_id=str(uuid.uuid4()), role=Role.ROLE_USER, parts=[part])
                    req = SendMessageRequest(
                        message=msg,
                        configuration=SendMessageConfiguration(return_immediately=True),
                    )
                    ctx = ClientCallContext(
                        state={},
                        service_parameters={"Authorization": token or ""},
                    )
                    envelope: dict[str, Any] = {"ok": False, "error": "UPSTREAM_ERROR", "message": "no response"}
                    async for sr in client_obj.send_message(req, context=ctx):
                        if sr.HasField("task"):
                            for art in sr.task.artifacts:
                                for p in art.parts:
                                    if p.data is not None:
                                        envelope = json_format.MessageToDict(p.data)
                    return envelope
                finally:
                    await client_obj.close()
        except Exception as e:
            return {"ok": False, "error": "UPSTREAM_ERROR", "message": str(e)}


a2a_client = OrdersA2AClient()


async def get_current_merchant() -> dict[str, Any] | None:
    token = get_token()
    if not token:
        return None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            headers = {"Authorization": token}
            store_url = os.environ.get("STORE_API_URL", "http://store-api:8002")
            resp = await client.get(f"{store_url}/auth/me", headers=headers)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


VALID_REFUND_FROM = {"PAID", "SHIPPED", "DELIVERED"}
VALID_CANCEL_FROM = {"PLACED", "PAID"}

_active_llm: Any = None


def set_active_llm(llm: Any) -> None:
    global _active_llm
    _active_llm = llm


async def agent_node(state: State, config: RunnableConfig, writer=None) -> dict[str, Any]:
    llm = config.get("configurable", {}).get("llm") or _active_llm
    if llm is None:
        from llm import get_llm
        from prompts import AGENT_TOOLS
        llm = get_llm().bind_tools(AGENT_TOOLS)

    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    cleaned = []
    for m in msgs:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                m = HumanMessage(content=redact_pii(content))
            else:
                m = HumanMessage(content=content)
        if isinstance(m, HumanMessage):
            cleaned.append(HumanMessage(content=redact_pii(getattr(m, "content", "") or "")))
        else:
            cleaned.append(m)
    resp = await llm.ainvoke(cleaned, config=config)
    content = getattr(resp, "content", "") or ""
    tool_calls: list[dict[str, Any]] = []
    for tc in getattr(resp, "tool_calls", []) or []:
        tool_calls.append(
            {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args", {}) or {}}
        )
    if tool_calls:
        ai = AIMessage(content=content, tool_calls=tool_calls)
    else:
        ai = AIMessage(content=content or "I can help with your orders.")
    return {"messages": [ai]}


async def read_tool(state: State, config: RunnableConfig) -> dict[str, Any]:
    token = get_token()
    tool_calls: list[dict[str, Any]] = []
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            tool_calls = list(getattr(m, "tool_calls", []) or [])
            break
    results: list[Any] = []
    for tc in tool_calls:
        name = tc.get("name") or tc.get("function", {}).get("name")
        args = tc.get("args") or {}
        if name not in ("list_orders", "lookup_order", "search_customer_orders"):
            continue
        envelope = await a2a_client.call_skill(name, args, token)
        data = envelope.get("data")
        if isinstance(data, list):
            envelope["data"] = [sanitize_order(o) if isinstance(o, dict) else o for o in data]
        elif isinstance(data, dict):
            envelope["data"] = sanitize_order(data)
        wrapped = untrusted_wrap(json.dumps(envelope, default=str))
        results.append(ToolMessage(content=wrapped, name=name, tool_call_id=tc.get("id") or str(uuid.uuid4())))
    return {"messages": results}


async def prepare_proposal(state: State, config: RunnableConfig) -> dict[str, Any]:
    token = get_token()
    tool_calls: list[dict[str, Any]] = []
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            tool_calls = list(getattr(m, "tool_calls", []) or [])
            break
    outputs: list[Any] = []
    proposal: dict[str, Any] | None = None
    for tc in tool_calls:
        name = tc.get("name") or tc.get("function", {}).get("name")
        args = tc.get("args") or {}
        if name not in ("propose_refund", "propose_cancel"):
            continue
        oid = args.get("id")
        action = "refund" if name == "propose_refund" else "cancel"
        envelope = await a2a_client.call_skill("lookup_order", {"id": oid}, token)
        if not envelope.get("ok"):
            outputs.append(ToolMessage(
                content=untrusted_wrap(json.dumps(envelope, default=str)),
                name=name,
                tool_call_id=tc.get("id") or str(uuid.uuid4()),
            ))
            continue
        order = sanitize_order(envelope.get("data") or {})
        current_status = order.get("status")
        valid_from = VALID_REFUND_FROM if action == "refund" else VALID_CANCEL_FROM
        proposed_status = "REFUNDED" if action == "refund" else "CANCELLED"
        if current_status not in valid_from:
            outputs.append(ToolMessage(
                content=untrusted_wrap(json.dumps({
                    "ok": False, "error": "INVALID_STATE",
                    "message": f"{action.title()} not allowed from status {current_status}.",
                }, default=str)),
                name=name,
                tool_call_id=tc.get("id") or str(uuid.uuid4()),
            ))
            continue
        merchant_info = await get_current_merchant() or {}
        proposal = {
            "proposal_id": str(uuid.uuid4()),
            "action": action,
            "order_id": oid,
            "merchant_id": merchant_info.get("merchant") or "",
            "amount": order.get("total"),
            "current_status": current_status,
            "proposed_status": proposed_status,
            "status": "PENDING",
        }
        outputs.append(ToolMessage(
            content="Proposal created, awaiting human approval.",
            name=name,
            tool_call_id=tc.get("id") or str(uuid.uuid4()),
        ))
    if proposal is not None:
        text = (
            f"{proposal['action'].title()} proposal created for order {proposal['order_id']}. "
            f"Amount: {proposal['amount']}. Current status: {proposal['current_status']}. "
            f"Proposed status: {proposal['proposed_status']}. "
            "Please approve or reject."
        )
        outputs.append(AIMessage(content=text))
    return {"messages": outputs, "pending_proposal": proposal}


async def approval(state: State, config: RunnableConfig) -> dict[str, Any]:
    proposal = state.get("pending_proposal")
    if not proposal:
        return {"messages": [AIMessage(content="No pending proposal to act on.")]}
    decision = interrupt(proposal)
    token = get_token()
    outputs: list[Any] = []
    if decision == "approve":
        envelope = await a2a_client.call_skill("lookup_order", {"id": proposal["order_id"]}, token)
        if not envelope.get("ok") or (envelope.get("data") or {}).get("status") != proposal["current_status"]:
            outputs.append(AIMessage(content=(
                f"Aborted: order {proposal['order_id']} status changed since the proposal "
                f"({proposal['current_status']}). No changes were made."
            )))
            return {"messages": outputs, "pending_proposal": None}
        skill = "refund_order" if proposal["action"] == "refund" else "cancel_order"
        result = await a2a_client.call_skill(skill, {"id": proposal["order_id"]}, token)
        if result.get("ok"):
            outputs.append(AIMessage(content=(
                f"{proposal['action'].title()} executed for order {proposal['order_id']}. "
                f"New status: {proposal['proposed_status']}."
            )))
        else:
            outputs.append(AIMessage(content=(
                f"Execution failed for order {proposal['order_id']}: "
                f"{result.get('error', 'unknown')} - {result.get('message', '')}"
            )))
    else:
        outputs.append(AIMessage(content="Rejected. No changes were made."))
    return {"messages": outputs, "pending_proposal": None}


def _has_tool_calls(messages: list[Any]) -> bool:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return bool(getattr(m, "tool_calls", None))
    return False


def route_after_agent(state: State) -> str:
    if _has_tool_calls(state["messages"]):
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                for tc in getattr(m, "tool_calls", []) or []:
                    name = tc.get("name") or tc.get("function", {}).get("name")
                    if name in ("propose_refund", "propose_cancel"):
                        return "prepare_proposal"
                return "read_tool"
    return END


def route_after_prepare(state: State) -> str:
    proposal = state.get("pending_proposal")
    if proposal is not None:
        return "approval"
    return "agent"
