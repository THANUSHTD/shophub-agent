"""R0: Mock store API tests - filter by status, tenant filtering, state
transitions, 401/403 auth, legacy lookup always 200."""

import pytest


@pytest.mark.asyncio
async def test_auth_me(client, store_url, token_a):
    resp = await client.get(f"{store_url}/auth/me", headers={"Authorization": f"Bearer {token_a}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user"] == "support_a"
    assert data["merchant"] == "merchant_a"


@pytest.mark.asyncio
async def test_auth_me_no_merchant(client, store_url, token_none):
    resp = await client.get(f"{store_url}/auth/me", headers={"Authorization": f"Bearer {token_none}"})
    assert resp.status_code == 200
    assert resp.json()["merchant"] is None


@pytest.mark.asyncio
async def test_auth_401_bad_token(client, store_url):
    resp = await client.get(f"{store_url}/auth/me", headers={"Authorization": "nope"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_401_missing(client, store_url):
    resp = await client.get(f"{store_url}/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_list_orders_status_filter(client, store_url, token_a):
    resp = await client.get(
        f"{store_url}/orders", params={"status": "PAID"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    orders = resp.json()
    assert all(o["merchant"] == "merchant_a" for o in orders)
    assert all(o["status"] == "PAID" for o in orders)
    ids = {o["id"] for o in orders}
    assert {"1042", "1043", "1045"}.issubset(ids)


@pytest.mark.asyncio
async def test_list_orders_invalid_status_422(client, store_url, token_a):
    resp = await client.get(
        f"{store_url}/orders", params={"status": "BOGUS"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_orders_tenant_isolation(client, store_url, token_a):
    resp = await client.get(
        f"{store_url}/orders", headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    orders = resp.json()
    assert all(o["merchant"] == "merchant_a" for o in orders)
    assert not any(o["id"] in {"2001", "2002"} for o in orders)


@pytest.mark.asyncio
async def test_list_orders_no_merchant_403(client, store_url, token_none):
    resp = await client.get(
        f"{store_url}/orders", headers={"Authorization": f"Bearer {token_none}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_lookup_always_200(client, store_url, token_a):
    # Found
    resp = await client.get(
        f"{store_url}/orders/lookup", params={"id": "1042"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] is True
    assert body["data"]["id"] == "1042"

    # Not found
    resp = await client.get(
        f"{store_url}/orders/lookup", params={"id": "9999"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] is False

    # Foreign order -> not found (never reveal existence)
    resp = await client.get(
        f"{store_url}/orders/lookup", params={"id": "2001"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] is False


@pytest.mark.asyncio
async def test_refund_valid_transitions(client, store_url, token_a):
    # 1042 is PAID -> refundable
    resp = await client.post(
        f"{store_url}/orders/refund",
        json={"id": "1042"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "REFUNDED"

    # Refund again -> 409 (already REFUNDED)
    resp = await client.post(
        f"{store_url}/orders/refund",
        json={"id": "1042"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_cancel_valid_transitions(client, store_url, token_a):
    # 1044 is PLACED -> cancelable
    resp = await client.post(
        f"{store_url}/orders/cancel",
        json={"id": "1044"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "CANCELLED"

    # Cancel again -> 409
    resp = await client.post(
        f"{store_url}/orders/cancel",
        json={"id": "1044"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_refund_foreign_order_404(client, store_url, token_a):
    # merchant_a tries to refund merchant_b's order 2001 -> 404 (identical to not found)
    resp = await client.post(
        f"{store_url}/orders/refund",
        json={"id": "2001"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_foreign_order_404(client, store_url, token_a):
    resp = await client.post(
        f"{store_url}/orders/cancel",
        json={"id": "2001"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_health(client, store_url):
    resp = await client.get(f"{store_url}/health")
    assert resp.status_code == 200
