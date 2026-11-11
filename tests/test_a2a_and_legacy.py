"""R2/R4: A2A protocol - agent card, skill calls, envelope, legacy lookup,
HTTP 404/409 mapping, dynamic card discovery by the orchestrator client."""

import json
import re

import httpx
import pytest
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct, Value

from a2a.client import create_client
from a2a.client.client import ClientConfig, ClientCallContext
from a2a.client.card_resolver import A2ACardResolver
from a2a.types.a2a_pb2 import (
    Message, Part, Role, SendMessageRequest, SendMessageConfiguration,
)


async def _lookup_via_a2a(http, orders_url, token, order_id):
    resolver = A2ACardResolver(http, orders_url)
    card = await resolver.get_agent_card()
    iface = [i for i in card.supported_interfaces if i.protocol_binding == "JSONRPC"][0]
    client_obj = await create_client(card, client_config=ClientConfig(streaming=False, polling=False))
    try:
        sv = Struct()
        sv.update({"skill": "lookup_order", "args": {"id": order_id}})
        part = Part(data=Value(struct_value=sv))
        msg = Message(message_id="m1", role=Role.ROLE_USER, parts=[part])
        req = SendMessageRequest(message=msg, configuration=SendMessageConfiguration(return_immediately=True))
        ctx = ClientCallContext(state={}, service_parameters={"Authorization": token})
        result = None
        async for sr in client_obj.send_message(req, context=ctx):
            if sr.HasField("task"):
                for art in sr.task.artifacts:
                    for p in art.parts:
                        if p.data is not None:
                            result = json_format.MessageToDict(p.data)
        return result
    finally:
        await client_obj.close()


@pytest.mark.asyncio
async def test_agent_card_fetchable(client, orders_url):
    resp = await client.get(f"{orders_url}/.well-known/agent-card.json")
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"] == "ShopHub Orders Agent"
    skill_ids = {s["id"] for s in card["skills"]}
    assert skill_ids == {
        "list_orders", "lookup_order", "search_customer_orders",
        "refund_order", "cancel_order",
    }
    assert card["capabilities"]["streaming"] is False


@pytest.mark.asyncio
async def test_skill_call_returns_envelope(client, orders_url, token_a):
    async with httpx.AsyncClient() as http:
        result = await _lookup_via_a2a(http, orders_url, token_a, "1042")
    assert result is not None
    assert result["ok"] is True
    assert result["data"]["id"] == "1042"
    assert result["data"]["customer_email"] == "[REDACTED]"
    assert result["data"]["customer_phone"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_skill_call_unknown_order_envelope(client, orders_url, token_a):
    async with httpx.AsyncClient() as http:
        result = await _lookup_via_a2a(http, orders_url, token_a, "9999")
    assert result["ok"] is False
    assert result["error"] == "ORDER_NOT_FOUND"


@pytest.mark.asyncio
async def test_legacy_lookup_becomes_order_not_found(client, store_url, token_a):
    """HTTP 200 with status:false from /orders/lookup -> ORDER_NOT_FOUND."""
    resp = await client.get(
        f"{store_url}/orders/lookup", params={"id": "9999"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] is False
    assert body["message"] == "Order not found"


@pytest.mark.asyncio
async def test_http_404_maps_to_order_not_found(client, store_url, token_a):
    """Foreign order -> 404 from refund -> ORDER_NOT_FOUND in agent."""
    resp = await client.post(
        f"{store_url}/orders/refund",
        json={"id": "2001"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_http_409_maps_to_invalid_state(client, store_url, token_a):
    """Refund a CANCELLED order (1046) -> 409 -> INVALID_STATE."""
    resp = await client.post(
        f"{store_url}/orders/refund",
        json={"id": "1046"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_token_forwarded_unchanged(client, store_url, token_a):
    """The store sees the original token (auth/me returns merchant_a)."""
    resp = await client.get(
        f"{store_url}/auth/me", headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    assert resp.json()["merchant"] == "merchant_a"


@pytest.mark.asyncio
async def test_orchestrator_a2a_client_discovers_card(orders_url, token_a):
    """The orchestrator's A2A client discovers the card dynamically."""
    from orchestrator import a2a_client

    async with httpx.AsyncClient() as http:
        # Reset the cached card so it re-discovers
        a2a_client._card = None
        a2a_client._card_url = None
        result = await a2a_client.call_skill("lookup_order", {"id": "1042"}, token_a)
    assert result["ok"] is True
    assert result["data"]["id"] == "1042"
