"""Debug utility for all three FastAPI services.

Runs a series of read-only checks against the running services and reports
PASS/FAIL for each. No state is modified.

Usage:
  python debug.py
  python debug.py --orchestrator http://localhost:8000 --orders http://localhost:8001 --store http://localhost:8002
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

import httpx

TOKEN_A = "token_merchant_a"
TOKEN_B = "token_merchant_b"
TOKEN_NONE = "token_no_merchant"


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _get(client: httpx.AsyncClient, url: str, headers: dict[str, str] | None = None):
    try:
        resp = await client.get(url, headers=headers)
        return resp.status_code, resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
    except Exception as e:
        return None, f"ERROR: {type(e).__name__}: {e}"


async def _post_json(client: httpx.AsyncClient, url: str, json: Any, headers: dict[str, str] | None = None):
    try:
        resp = await client.post(url, json=json, headers=headers)
        return resp.status_code, resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
    except Exception as e:
        return None, f"ERROR: {type(e).__name__}: {e}"


async def check_health(client: httpx.AsyncClient, name: str, url: str) -> bool:
    status, body = await _get(client, f"{url}/health")
    ok = status == 200
    print(f"[{'PASS' if ok else 'FAIL'}] {name}/health -> {status} {body if not ok else ''}")
    return ok


async def check_auth_me(client: httpx.AsyncClient, store_url: str, token: str, label: str, expected_merchant: str | None) -> bool:
    status, body = await _get(client, f"{store_url}/auth/me", _auth_headers(token))
    ok = (
        status == 200
        and isinstance(body, dict)
        and body.get("merchant") == expected_merchant
    )
    print(f"[{'PASS' if ok else 'FAIL'}] /auth/me {label} -> {status} merchant={body.get('merchant') if isinstance(body, dict) else body}")
    return ok


async def check_auth_401(client: httpx.AsyncClient, store_url: str) -> bool:
    status, _ = await _get(client, f"{store_url}/auth/me", {"Authorization": "Bearer bad-token"})
    ok = status == 401
    print(f"[{'PASS' if ok else 'FAIL'}] /auth/me bad token -> {status} (expected 401)")
    return ok


async def check_auth_403(client: httpx.AsyncClient, store_url: str) -> bool:
    status, _ = await _get(client, f"{store_url}/orders", _auth_headers(TOKEN_NONE))
    ok = status == 403
    print(f"[{'PASS' if ok else 'FAIL'}] /orders no-merchant token -> {status} (expected 403)")
    return ok


async def check_list_orders(client: httpx.AsyncClient, store_url: str, token: str, label: str, expected_ids: set[str]) -> bool:
    status, body = await _get(client, f"{store_url}/orders", _auth_headers(token))
    ids = {o["id"] for o in body} if isinstance(body, list) else set()
    ok = status == 200 and expected_ids.issubset(ids)
    print(f"[{'PASS' if ok else 'FAIL'}] /orders {label} -> {status} ids={sorted(ids)}")
    return ok


async def check_lookup(client: httpx.AsyncClient, store_url: str, token: str, oid: str, label: str, expect_found: bool) -> bool:
    status, body = await _get(client, f"{store_url}/orders/lookup", _auth_headers(token), params={"id": oid})
    found = isinstance(body, dict) and body.get("status") is True
    ok = status == 200 and found == expect_found
    print(f"[{'PASS' if ok else 'FAIL'}] /orders/lookup?id={oid} {label} -> {status} found={found}")
    return ok


async def check_agent_card(client: httpx.AsyncClient, orders_url: str) -> bool:
    status, body = await _get(client, f"{orders_url}/.well-known/agent-card.json")
    skills = {s["id"] for s in body.get("skills", [])} if isinstance(body, dict) else set()
    expected = {"list_orders", "lookup_order", "search_customer_orders", "refund_order", "cancel_order"}
    ok = status == 200 and expected.issubset(skills) and body.get("capabilities", {}).get("streaming") is False
    print(f"[{'PASS' if ok else 'FAIL'}] agent card -> {status} skills={sorted(skills)} streaming={body.get('capabilities', {}).get('streaming') if isinstance(body, dict) else 'N/A'}")
    return ok


async def check_cross_merchant(client: httpx.AsyncClient, store_url: str) -> bool:
    # Merchant A looks up merchant B's order -> not found
    status, body = await _get(client, f"{store_url}/orders/lookup", _auth_headers(TOKEN_A), params={"id": "2001"})
    found = isinstance(body, dict) and body.get("status") is True
    ok = status == 200 and found is False
    print(f"[{'PASS' if ok else 'FAIL'}] cross-merchant lookup 2001 by A -> {status} found={found} (expected False)")
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser(description="Debug ShopHub FastAPI services")
    parser.add_argument("--orchestrator", default="http://localhost:8000")
    parser.add_argument("--orders", default="http://localhost:8001")
    parser.add_argument("--store", default="http://localhost:8002")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    results: list[bool] = []
    timeout = httpx.Timeout(args.timeout)

    async with httpx.AsyncClient(timeout=timeout) as client:
        print("=== Health checks ===")
        results.append(await check_health(client, "store-api", args.store))
        results.append(await check_health(client, "orders-agent", args.orders))
        results.append(await check_health(client, "orchestrator", args.orchestrator))

        print("\n=== Auth checks ===")
        results.append(await check_auth_me(client, args.store, TOKEN_A, "merchant_a", "merchant_a"))
        results.append(await check_auth_me(client, args.store, TOKEN_B, "merchant_b", "merchant_b"))
        results.append(await check_auth_me(client, args.store, TOKEN_NONE, "no_merchant", None))
        results.append(await check_auth_401(client, args.store))
        results.append(await check_auth_403(client, args.store))

        print("\n=== Tenant isolation ===")
        results.append(await check_list_orders(client, args.store, TOKEN_A, "merchant_a", {"1040", "1041", "1042", "1043", "1044", "1045", "1046"}))
        results.append(await check_list_orders(client, args.store, TOKEN_B, "merchant_b", {"2001", "2002", "2003", "2004", "2005", "2006", "2007"}))
        results.append(await check_lookup(client, args.store, TOKEN_A, "1042", "merchant_a own order", True))
        results.append(await check_lookup(client, args.store, TOKEN_A, "2001", "merchant_b order by A", False))
        results.append(await check_cross_merchant(client, args.store))

        print("\n=== A2A agent card ===")
        results.append(await check_agent_card(client, args.orders))

    print(f"\n=== Result: {sum(results)}/{len(results)} checks passed ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
