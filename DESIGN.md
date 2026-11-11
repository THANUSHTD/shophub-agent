# ShopHub Agent - Design Document

## Problem and scope

ShopHub hosts many independent merchant stores (tenants). Merchant support
staff chat with an AI assistant that answers questions about their store's
orders and can PROPOSE refunds or cancellations. Key constraints:

- Tenant isolation: a merchant must never see another merchant's data.
- Human-in-the-loop: refunds/cancellations never execute without explicit
  approval.
- Order notes are untrusted and may contain prompt injection.
- Customer emails/phones must never reach the LLM.
- Orchestrator and orders sub-agent are separate processes talking only over
  Google's A2A protocol (JSON-RPC 2.0 over HTTP, discovered via Agent Card).

Out of scope: web UI, registration, OAuth, real payments, RAG/vector stores,
multiple LLM providers, multiple sub-agents, Kubernetes, A2A streaming/push,
A2A task store (in-memory only).

## Architecture

```
+----------------+      A2A JSON-RPC 2.0      +------------------+
|                |                            |                  |
|  Orchestrator  |============================>|  Orders Agent    |
|  port 8000     |   Agent Card + JSON-RPC      |  port 8001       |
|                |   /.well-known/agent-card.json|                  |
+----------------+                            +------------------+
        ^                                               ^
        | HTTP                                            | HTTP
        |                                               v
+----------------+                            +------------------+
|                |                            |                  |
|   Store API    |============================>|   Postgres       |
|  port 8002     |   (mock, in-memory)          |  (checkpoints)   |
|                |                            |                  |
+----------------+                            +------------------+
```

The orchestrator and orders agent are separate processes. They discover each
other via the Agent Card at `/.well-known/agent-card.json` and communicate
over JSON-RPC 2.0. The orchestrator never imports from `orders_agent.py`;
they share only `guardrails.py` and `config.py`.

## Flow of one turn

1. Client POSTs `/chat/stream` with `Authorization: Bearer <token>`.
2. Orchestrator calls `GET /auth/me` on the Store API to resolve the merchant.
   `token_no_merchant` -> HTTP 403 (fail closed).
3. Thread ID is namespaced: `{merchant}:{thread_id}`. Another merchant can never
   resume the thread.
4. LangGraph runs:
   - `agent`: LLM (FakeChatModel or ChatOpenAI) with 5 bound tools:
     `list_orders`, `lookup_order`, `search_customer_orders`,
     `propose_refund`, `propose_cancel`. No execute tool exists.
   - `read_tool`: for read skills, calls Orders Agent over A2A, sanitises
     result, wraps with "UNTRUSTED DATA..." prefix, returns ToolMessage.
   - `prepare_proposal`: for `propose_*`, calls `lookup_order` via A2A,
     validates state transition, builds a PENDING proposal, returns a
     templated AI message, then `interrupt(proposal)` pauses.
   - `approval`: pauses for human decision. On approve, re-checks status
     (abort if changed), then calls `refund_order`/`cancel_order` via A2A.
     On reject, no execute skill is called.
5. SSE events stream in real time: `tool_started`, `tool_finished`,
   `token` (real LLM chunks), `proposal` (on interrupt), `error`, `done`.
6. `/chat/approve` and `/chat/reject` resume the graph with
   `Command(resume="approve"|"reject")`. Double approval -> HTTP 409.

## A2A usage

The Orders Agent serves an Agent Card at `/.well-known/agent-card.json`
(name, description, version, capabilities.streaming=false, bearer security
scheme, 5 skills). The Orchestrator's `OrdersA2AClient` lazily resolves the
card via `A2ACardResolver`, reads the JSON-RPC interface URL from
`supportedInterfaces`, and sends `message/send` with a DataPart
`{"skill": "<id>", "args": {...}}`. The response is a completed Task with a
DataPart artifact containing `{"ok":..., "data":...}` or
`{"ok":false, "error":..., "message":...}`.

The agent is deterministic: no LLM, no streaming, in-memory task store.

## Credential forwarding and tenant isolation

- The bearer token is read ONLY from the incoming HTTP `Authorization` header
  (SDK populates `ServerCallContext.state["headers"]`). It is never read from
  message text/metadata and never cached or logged.
- The token is forwarded unchanged to the Store API as the `Authorization`
  header. The "Bearer " prefix is stripped on receipt and re-added on forward
  so the Store API's `HTTPBearer` dependency works correctly.
- The token is held in a `contextvars.ContextVar` for the duration of the
  request. It is NEVER in LangGraph state, config, prompts, or messages.
- Checkpoint thread ID is `{merchant}:{thread_id}`, so a thread can never be
  resumed by another merchant.
- Every Store API call is scoped to the caller's merchant; foreign orders
  return identical 404/200+status:false responses.

## Guardrails

- `redact_pii`: regex-redacts emails and 10+ digit phones. Leaves 4-digit
  order ids and totals untouched.
- `neutralize_note`: detects instruction-like patterns ("ignore all
  instructions", "assistant:", "system:", "admin mode", "refund this order",
  "cancel all orders", etc.) and replaces the whole note with a neutralised
  placeholder, flagging it.
- `sanitize_order`: whitelist - id, merchant, customer_name, status, total,
  placed_at, neutralised note, note_flagged; email/phone always "[REDACTED]".
- `untrusted_wrap`: prefixes tool results with "UNTRUSTED DATA from a tool...".
- `require_merchant`: raises `NoMerchantError` for null merchant (fail closed).
- Applied in defence-in-depth: orders agent sanitises before returning;
  orchestrator sanitises again (idempotent) before tool results reach the model.

## Human approval

The model has NO tool that executes a refund or cancel. It can only call
`propose_refund` / `propose_cancel`, which creates a PENDING proposal and
interrupts. Execution happens only in the `approval` node after human
`Command(resume="approve")`. On approve, status is re-checked (abort if
changed) before calling the execute skill. Double approval returns 409.
Reject never calls any execute skill.

## SSE

`POST /chat/stream` returns `text/event-stream`. Events are produced by
LangGraph's `astream` with `stream_mode=["updates","custom","messages"]`:
`tool_started`, `tool_finished`, `token` (real LLM chunks), `proposal`
(on interrupt), `error`, `done`.

## Persistence

`CHECKPOINTER=postgres` uses `AsyncPostgresSaver.from_conn_string()`
(autocommit, dict_row) with `.setup()` in FastAPI lifespan.
`CHECKPOINTER=memory` uses `InMemorySaver` (tests/evals).
State including pending proposals survives orchestrator restarts.

## Fake LLM and testing

`FakeChatModel` (langchain `BaseChatModel`) decides tool calls from regex on
the user message, streams real chunks, and records every prompt in
`seen_prompts`. `obedient=True` (tests/evals only) also emits
`propose_refund` when it sees "refund" in tool output - making the injection
test meaningful (worst case is a pending proposal, never execution).

## Key trade-offs

- In-memory A2A task store (no durable task persistence for the agent).
- Deterministic fake LLM by default; OpenAI optional via env var.
- No web UI, OAuth, payments, RAG, or vector store.
- Single sub-agent (orders), not multiple.

## What was cut and why

RAG/vector stores, multiple LLM providers, multiple sub-agents, payments,
Kubernetes, A2A streaming/push, web UI, registration/OAuth - all explicitly
out of scope per the assignment. A2A task store is in-memory because the
orders agent is deterministic and stateless per request.

## Cost and latency

No real OpenAI key is configured by default, so per-turn cost is $0 with the
fake LLM. Latency is sub-millisecond for deterministic paths (guardrails,
store lookups) and a few milliseconds for the A2A round trip. With OpenAI,
latency and cost depend on the model (`gpt-4.1-mini`); actual pricing must be
consulted from OpenAI's published rates (not estimated here).

## One more week

- Add A2A streaming (`message/stream`) and push notifications.
- Add durable task store for the orders agent (Postgres-backed).
- Add OpenAI traces and latency/cost measurement dashboard.
- Add respx-based contract tests for the A2A client.
- Add rate limiting and request ID tracing.

---

Built with the assistance of an AI coding assistant. I reviewed and can
explain every line of this code.
