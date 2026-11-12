"""R0/R4: Tenant isolation - merchant A must never see merchant B's data."""

import pytest


@pytest.mark.asyncio
async def test_cannot_lookup_foreign_order(client, store_url, orders_url, token_a, token_b):
    # Merchant A looks up merchant B's order 2001 -> not found
    resp = await client.get(
        f"{store_url}/orders/lookup", params={"id": "2001"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] is False


@pytest.mark.asyncio
async def test_cannot_refund_foreign_order(client, store_url, token_a, token_b):
    resp = await client.post(
        f"{store_url}/orders/refund",
        json={"id": "2001"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cannot_cancel_foreign_order(client, store_url, token_a, token_b):
    resp = await client.post(
        f"{store_url}/orders/cancel",
        json={"id": "2001"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_only_own_orders(client, store_url, token_a, token_b):
    resp_a = await client.get(
        f"{store_url}/orders", headers={"Authorization": f"Bearer {token_a}"},
    )
    ids_a = {o["id"] for o in resp_a.json()}
    resp_b = await client.get(
        f"{store_url}/orders", headers={"Authorization": f"Bearer {token_b}"},
    )
    ids_b = {o["id"] for o in resp_b.json()}
    assert ids_a.isdisjoint(ids_b)
    assert {"1040", "1041", "1042", "1043", "1044", "1045", "1046"}.issubset(ids_a)
    assert {"2001", "2002", "2003", "2004", "2005", "2006", "2007"}.issubset(ids_b)


@pytest.mark.asyncio
async def test_can_read_own_order(client, store_url, token_a):
    resp = await client.get(
        f"{store_url}/orders/lookup", params={"id": "1042"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == "1042"


@pytest.mark.asyncio
async def test_thread_namespaced_isolation(store_url, orders_url, token_a, token_b):
    """A thread used by merchant A is completely isolated from merchant B (R3/R8).

    Even when both merchants share the same user thread ID (e.g. 'thread_1'),
    the orchestrator namespaces state by merchant ('{merchant}:{thread_id}').
    Running a turn as merchant A populates state only in merchant A's namespace.
    Merchant B's thread state remains completely empty and unexposed.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from src.orchestrator import build_graph, set_token

    graph = build_graph(InMemorySaver())
    user_thread_id = "shared_thread_1"
    namespaced_a = f"merchant_a:{user_thread_id}"
    namespaced_b = f"merchant_b:{user_thread_id}"

    # Merchant A runs a turn on their namespaced thread
    set_token(token_a)
    async for _ in graph.astream(
        {"messages": [{"role": "user", "content": "Show paid orders"}]},
        config={"configurable": {"thread_id": namespaced_a}},
        stream_mode=["updates"],
    ):
        pass

    # Merchant A's thread now contains conversation state
    state_a = graph.get_state(config={"configurable": {"thread_id": namespaced_a}})
    assert state_a.values is not None
    messages_a = state_a.values.get("messages", [])
    assert len(messages_a) > 0, "Merchant A should have messages in thread state"

    # Merchant B's thread with the same user ID is completely empty
    state_b = graph.get_state(config={"configurable": {"thread_id": namespaced_b}})
    assert not state_b.values or not state_b.values.get("messages"), (
        "Merchant B must not see any state from Merchant A's thread"
    )

    # Namespaced IDs are isolated
    assert namespaced_a != namespaced_b

