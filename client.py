"""SSE demo client for the ShopHub orchestrator.

Usage:
  python client.py chat --token token_merchant_a --thread t1 "Show paid orders"
  python client.py approve --token token_merchant_a --thread t1 --proposal <id>
  python client.py reject --token token_merchant_a --thread t1 --proposal <id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

ORCHESTRATOR_URL = "http://127.0.0.1:8000"


async def chat(token: str, thread: str, message: str):
    url = f"{ORCHESTRATOR_URL}/chat/stream"
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", url,
            json={"thread_id": thread, "message": message},
            headers={"Authorization": token},
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    event = data.get("event", "message")
                    payload = data.get("data", {})
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    print(f"[{event}] {json.dumps(payload)}")


async def approve(token: str, thread: str, proposal_id: str):
    await _decision(token, thread, proposal_id, "approve")


async def reject(token: str, thread: str, proposal_id: str):
    await _decision(token, thread, proposal_id, "reject")


async def _decision(token, thread, proposal_id, decision):
    url = f"{ORCHESTRATOR_URL}/chat/{decision}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            json={"thread_id": thread, "proposal_id": proposal_id},
            headers={"Authorization": token},
        )
        print(json.dumps(resp.json(), indent=2))


def main():
    parser = argparse.ArgumentParser(description="ShopHub SSE demo client")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="Stream a chat message")
    p_chat.add_argument("--token", required=True)
    p_chat.add_argument("--thread", required=True)
    p_chat.add_argument("message")

    p_ap = sub.add_parser("approve", help="Approve a pending proposal")
    p_ap.add_argument("--token", required=True)
    p_ap.add_argument("--thread", required=True)
    p_ap.add_argument("--proposal", required=True)

    p_rj = sub.add_parser("reject", help="Reject a pending proposal")
    p_rj.add_argument("--token", required=True)
    p_rj.add_argument("--thread", required=True)
    p_rj.add_argument("--proposal", required=True)

    args = parser.parse_args()
    if args.command == "chat":
        asyncio.run(chat(args.token, args.thread, args.message))
    elif args.command == "approve":
        asyncio.run(approve(args.token, args.thread, args.proposal))
    elif args.command == "reject":
        asyncio.run(reject(args.token, args.thread, args.proposal))


if __name__ == "__main__":
    main()
