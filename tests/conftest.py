"""Pytest conftest: spins up the real store API and orders agent as uvicorn
servers on free ports in background threads, and builds the orchestrator
graph with InMemorySaver + FakeChatModel(obedient=True).

No API key, no network beyond localhost, fully deterministic.
"""

from __future__ import annotations

import asyncio
import json
import os

# Unit tests run with deterministic FakeChatModel (no network calls, fully deterministic)
os.environ["LLM_MODE"] = "fake"

import sys
from pathlib import Path
import threading
from typing import Any

# Ensure workspace root and src/ are on sys.path for test discovery
_root_dir = str(Path(__file__).resolve().parent.parent)
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
for _p in (_root_dir, _src_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import httpx
import pytest
import uvicorn

from src.config import MERCHANT_TOKENS
from src.guardrails import sanitize_order
from src.llm import FakeChatModel
from src.orchestrator import create_app, build_graph, a2a_client, _token_var
from src.orders_agent import create_app as create_orders_app
from src.store_api import create_app as create_store_app


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for(url: str, client: httpx.AsyncClient, timeout: float = 15.0):
    import time

    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return
        except Exception as e:
            last_err = e
        await asyncio.sleep(0.2)
    raise RuntimeError(f"Service {url} did not become healthy: {last_err}")


class _UVicornThread:
    def __init__(self, app, port: int):
        self._server = None
        self._thread = threading.Thread(
            target=self._run, args=(app, port), daemon=True
        )

    def _run(self, app, port: int):
        # Windows: use selector event loop policy so httpx/psycopg work in
        # background threads (Proactor loop can break localhost connections).
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(config)
        self._server = server
        loop.run_until_complete(server.serve())

    def start(self):
        self._thread.start()

    async def stop(self):
        if self._server:
            self._server.should_exit = True


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def store_url():
    port = _free_port()
    app = create_store_app()
    server = _UVicornThread(app, port)
    server.start()
    url = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient() as client:
        await _wait_for(f"{url}/health", client)
    os.environ["STORE_API_URL"] = url
    yield url


@pytest.fixture(scope="module")
async def orders_url(store_url):
    port = _free_port()
    os.environ["ORDERS_AGENT_PUBLIC_URL"] = f"http://127.0.0.1:{port}"
    os.environ["ORDERS_AGENT_URL"] = f"http://127.0.0.1:{port}"
    app = create_orders_app()
    server = _UVicornThread(app, port)
    server.start()
    url = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient() as client:
        await _wait_for(f"{url}/health", client)
    yield url


@pytest.fixture(scope="module")
def base_urls(store_url, orders_url):
    return {"store": store_url, "orders": orders_url}


@pytest.fixture(scope="module")
def graph():
    """Orchestrator graph with InMemorySaver + obedient FakeChatModel."""
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    llm = FakeChatModel(obedient=True).bind_tools([])
    return build_graph(checkpointer)


@pytest.fixture
async def client(base_urls):
    async with httpx.AsyncClient() as c:
        yield c


@pytest.fixture
def token_a():
    return "token_merchant_a"


@pytest.fixture
def token_b():
    return "token_merchant_b"


@pytest.fixture
def token_none():
    return "token_no_merchant"


async def _auth_me(client: httpx.AsyncClient, store_url: str, token: str):
    resp = await client.get(f"{store_url}/auth/me", headers={"Authorization": token})
    return resp


async def _me(client: httpx.AsyncClient, store_url: str, token: str) -> dict[str, Any] | None:
    resp = await _auth_me(client, store_url, token)
    if resp.status_code == 200:
        return resp.json()
    return None
