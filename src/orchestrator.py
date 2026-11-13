from __future__ import annotations
import contextlib
import json
import os
import httpx
from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
import config as _config
from guardrails import NoMerchantError
from graph import build_graph, State  # noqa: F401
from nodes import (  # noqa: F401
    a2a_client,
    set_token,
    get_token,
    _token_var,
    require_merchant,
)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if _config.CHECKPOINTER == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        try:
            async with AsyncPostgresSaver.from_conn_string(_config.DATABASE_URL) as saver:
                await saver.setup()
                app.state.graph = build_graph(saver)
                yield
        except Exception as e:
            raise RuntimeError(
                f"\n\n❌ [ShopHub Orchestrator] Failed to connect to PostgreSQL at '{_config.DATABASE_URL}'.\n"
                f"   Ensure Postgres is running via 'docker compose up -d postgres'\n"
                f"   or switch to in-memory mode by running with 'CHECKPOINTER=memory'.\n"
            ) from e
    else:
        app.state.graph = build_graph(InMemorySaver())
        yield


bearer_scheme = HTTPBearer(auto_error=False)


class ChatStreamRequest(BaseModel):
    thread_id: str = Field(..., description="Unique thread identifier (e.g. t1)", examples=["t1"])
    message: str = Field(..., description="User message to the assistant", examples=["Show paid orders"])


class DecisionRequest(BaseModel):
    thread_id: str = Field(..., description="Thread identifier (e.g. t1)", examples=["t1"])
    proposal_id: str = Field(..., description="Proposal UUID to approve or reject")


async def get_authenticated_merchant(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> tuple[str, str]:
    auth = (f"Bearer {credentials.credentials}" if credentials else None) or request.headers.get("authorization")
    if not auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    me = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            store_url = os.environ.get("STORE_API_URL", "http://store-api:8002")
            resp = await client.get(
                f"{store_url}/auth/me",
                headers={"Authorization": auth},
            )
            if resp.status_code == 200:
                me = resp.json()
    except Exception:
        pass
    if me is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    try:
        require_merchant(me)
    except NoMerchantError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No merchant scoped to token")
    return auth, me["merchant"]


def create_app(checkpointer=None, lifespan=lifespan) -> FastAPI:
    app_lifespan = None if checkpointer is not None else lifespan
    app = FastAPI(title="ShopHub Orchestrator", lifespan=app_lifespan)
    if checkpointer is not None:
        app.state.graph = build_graph(checkpointer)
    else:
        app.state.graph = build_graph(InMemorySaver())

    def _get_graph():
        return getattr(app.state, "graph", None) or build_graph(checkpointer)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/chat/stream", summary="Stream chat message via SSE")
    async def chat_stream(
        body: ChatStreamRequest,
        auth_info: tuple[str, str] = Depends(get_authenticated_merchant),
    ):
        token, merchant = auth_info
        thread_id = body.thread_id
        message = body.message
        if not thread_id or not message:
            raise HTTPException(status_code=400, detail="thread_id and message required")

        namespaced_thread = f"{merchant}:{thread_id}"
        set_token(token)
        graph = _get_graph()

        async def event_gen():
            try:
                async for chunk in graph.astream(
                    {"messages": [HumanMessage(content=message)]},
                    config={"configurable": {"thread_id": namespaced_thread}},
                    stream_mode=["updates", "custom", "messages"],
                ):
                    kind, data = chunk if isinstance(chunk, tuple) else ("updates", chunk)
                    if kind == "messages":
                        msg, meta = data
                        if isinstance(msg, AIMessageChunk) and msg.content:
                            yield {"event": "token", "data": json.dumps({"content": msg.content})}
                    elif kind == "custom":
                        yield {"event": "custom", "data": json.dumps(data)}
                    elif kind == "updates":
                        if isinstance(data, dict):
                            for node_name, node_update in data.items():
                                if isinstance(node_update, dict):
                                    for m in node_update.get("messages") or []:
                                        if isinstance(m, ToolMessage):
                                            yield {"event": "tool_finished", "data": json.dumps({
                                                "tool": m.name, "content": m.content,
                                            })}
                                        elif isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                                            for tc in m.tool_calls:
                                                yield {"event": "tool_started", "data": json.dumps({
                                                    "tool": tc.get("name"), "args": tc.get("args"),
                                                })}
                state = await graph.aget_state(config={"configurable": {"thread_id": namespaced_thread}})
                if state.next and state.next[0] == "approval":
                    proposal = state.values.get("pending_proposal") or {}
                    yield {"event": "proposal", "data": json.dumps({
                        "proposal": proposal,
                        "text": f"{proposal.get('action','').title()} proposal created for order {proposal.get('order_id')}. Please approve or reject.",
                    })}
                yield {"event": "done", "data": json.dumps({"status": "complete"})}
            except Exception as e:
                yield {"event": "error", "data": json.dumps({"message": str(e)})}

        return EventSourceResponse(event_gen())

    @app.post("/chat/approve", summary="Approve a pending proposal")
    async def chat_approve(
        body: DecisionRequest,
        auth_info: tuple[str, str] = Depends(get_authenticated_merchant),
    ):
        return await _decision(body, auth_info, "approve")

    @app.post("/chat/reject", summary="Reject a pending proposal")
    async def chat_reject(
        body: DecisionRequest,
        auth_info: tuple[str, str] = Depends(get_authenticated_merchant),
    ):
        return await _decision(body, auth_info, "reject")

    async def _decision(
        body: DecisionRequest,
        auth_info: tuple[str, str],
        decision: str,
    ):
        token, merchant = auth_info
        thread_id = body.thread_id
        proposal_id = body.proposal_id
        if not thread_id or not proposal_id:
            raise HTTPException(status_code=400, detail="thread_id and proposal_id required")

        namespaced_thread = f"{merchant}:{thread_id}"
        set_token(token)
        graph = _get_graph()

        state = await graph.aget_state(config={"configurable": {"thread_id": namespaced_thread}})
        if state is None or state.values is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        proposal = state.values.get("pending_proposal")
        if not proposal or proposal.get("proposal_id") != proposal_id:
            raise HTTPException(status_code=404, detail="Proposal not found")
        if proposal.get("merchant_id") != merchant:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Proposal belongs to another merchant")
        if proposal.get("status") != "PENDING":
            raise HTTPException(status_code=409, detail="Proposal already resolved")

        try:
            async for _ in graph.astream(
                Command(resume=decision),
                config={"configurable": {"thread_id": namespaced_thread}},
                stream_mode=["updates"],
            ):
                pass
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        state2 = await graph.aget_state(config={"configurable": {"thread_id": namespaced_thread}})
        messages = state2.values.get("messages", []) if state2.values else []
        last_text = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and getattr(m, "content", ""):
                last_text = m.content
                break
        return {"status": "ok", "message": last_text or f"{decision}d"}

    @app.get("/chat/{thread_id}/messages", summary="Get thread message history")
    async def chat_messages(
        thread_id: str,
        auth_info: tuple[str, str] = Depends(get_authenticated_merchant),
    ):
        token, merchant = auth_info
        namespaced_thread = f"{merchant}:{thread_id}"
        set_token(token)
        graph = _get_graph()
        state = await graph.aget_state(config={"configurable": {"thread_id": namespaced_thread}})
        messages = state.values.get("messages", []) if state and state.values else []
        out = []
        for m in messages:
            out.append({
                "type": m.__class__.__name__,
                "content": getattr(m, "content", ""),
                "tool_calls": getattr(m, "tool_calls", None) or None,
                "name": getattr(m, "name", None) or None,
            })
        return {"messages": out}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ORCHESTRATOR_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
