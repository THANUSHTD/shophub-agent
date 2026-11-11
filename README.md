# ShopHub Senior AI Agent Engineer - Take-Home Solution

A multi-tenant ShopHub support assistant built with FastAPI, LangGraph, and
Google's A2A protocol (a2a-sdk 1.3.0, A2A spec 1.0). Merchant support staff
chat with an AI assistant that answers questions about their store's orders
and can PROPOSE refunds or cancellations - never execute them without explicit
human approval.

## Architecture

```
+----------------+      A2A (JSON-RPC 2.0)      +------------------+
|                |                              |                  |
|  Orchestrator  |<============================>|  Orders Agent    |
|  (port 8000)   |   Agent Card + JSON-RPC      |  (port 8001)     |
|                |   /.well-known/agent-card.json|                  |
+----------------+                              +------------------+
        ^                                               ^
        |                                               | HTTP (store API)
        |                                               v
+----------------+                              +------------------+
|                |                              |                  |
|   Store API    |<============================>|   Postgres       |
|  (port 8002)   |   (mock, in-memory)          |  (checkpoints)   |
|                |                              |                  |
+----------------+                              +------------------+
```

- **Orchestrator** (`orchestrator.py`): LangGraph graph with SSE chat API,
  human approval flow, and Postgres persistence. Talks to the Orders Agent
  ONLY over A2A.
- **Orders Agent** (`orders_agent.py`): Deterministic A2A server (no LLM).
  Exposes 5 skills over JSON-RPC 2.0. Forwards bearer tokens to the Store API.
- **Store API** (`store_api.py`): Mock multi-tenant backend with seed data.
- **Postgres**: LangGraph checkpointer for state persistence.

## Stack

| Component | Version |
|-----------|---------|
| Python | 3.12.3 |
| FastAPI | 0.162.4 |
| uvicorn | 0.54.0 |
| httpx | 0.28.1 |
| langgraph | 1.2.20 |
| langchain-openai | 1.7.4 |
| langgraph-checkpoint-postgres | 3.2.0 |
| psycopg[binary,pool] | 3.3.8 |
| a2a-sdk | 1.3.0 (A2A spec 1.0) |
| pytest | 9.2.0 |
| pytest-asyncio | 1.4.1.post1 |
| respx | 0.23.1 |

LLM provider: **OpenAI** (`gpt-4.1-mini`). Default mode is `fake` (deterministic
scripted model, no API key needed). Set `LLM_MODE=openai` and
`OPENAI_API_KEY` to use the real OpenAI API.

## Prerequisites

- Python 3.12.3
- Docker Desktop (for `docker compose up`)
- A PowerShell terminal

## Setup

1. Clone the repository.
2. Copy the environment example:

```powershell
Copy-Item .env.example .env
```

3. Edit `.env` if needed (defaults work for local Docker).
4. Create and activate the virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Running with Docker Compose (recommended)

```powershell
docker compose up --build
```

This starts `postgres`, `store-api`, `orders-agent`, and `orchestrator`.
Service URLs:

- Orchestrator: http://localhost:8000
- Orders Agent: http://localhost:8001
- Store API: http://localhost:8002
- Postgres: localhost:5433

## Running locally without Docker

Start each service in a separate terminal (ports 8002, 8001, 8000):

**Terminal 1 - Store API:**
```powershell
.venv\Scripts\Activate.ps1
$env:STORE_API_PORT="8002"
uvicorn store_api:app --host 127.0.0.1 --port 8002
```

**Terminal 2 - Orders Agent:**
```powershell
.venv\Scripts\Activate.ps1
$env:STORE_API_URL="http://127.0.0.1:8002"
$env:ORDERS_AGENT_URL="http://127.0.0.1:8001"
$env:ORDERS_AGENT_PUBLIC_URL="http://127.0.0.1:8001"
uvicorn orders_agent:app --host 127.0.0.1 --port 8001
```

**Terminal 3 - Orchestrator:**
```powershell
.venv\Scripts\Activate.ps1
$env:STORE_API_URL="http://127.0.0.1:8002"
$env:ORDERS_AGENT_URL="http://127.0.0.1:8001"
$env:ORDERS_AGENT_PUBLIC_URL="http://127.0.0.1:8001"
uvicorn orchestrator:app --host 127.0.0.1 --port 8000
```

## Running tests

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
```

## Running evals

```powershell
.venv\Scripts\Activate.ps1
python evals.py
```

Use `--real` to test with OpenAI (requires `OPENAI_API_KEY`):

```powershell
python evals.py --real
```

## Mock tokens (fixtures only)

| Token | User | Merchant |
|-------|------|----------|
| `token_merchant_a` | support_a | merchant_a |
| `token_merchant_b` | support_b | merchant_b |
| `token_no_merchant` | support_none | null |

## SSE demo client

```powershell
# Show paid orders for merchant A
python client.py chat --token token_merchant_a --thread t1 "Show paid orders"

# Approve a pending proposal
python client.py approve --token token_merchant_a --thread t1 --proposal <proposal-id>

# Reject a pending proposal
python client.py reject --token token_merchant_a --thread t1 --proposal <proposal-id>
```

## Agent Card

Fetch the Orders Agent card:

```powershell
# PowerShell (Invoke-RestMethod)
Invoke-RestMethod http://localhost:8001/.well-known/agent-card.json | ConvertTo-Json -Depth 10

# curl.exe (not the curl alias)
curl.exe http://localhost:8001/.well-known/agent-card.json
```

## Raw A2A JSON-RPC example

Send a `lookup_order` skill call to the Orders Agent:

```powershell
$body = @{
    jsonrpc = "2.0"
    id = "1"
    method = "SendMessage"
    params = @{
        message = @{
            messageId = "m1"
            role = "user"
            parts = @(
                @{
                    data = @{
                        skill = "lookup_order"
                        args = @{ id = "1042" }
                    }
                }
            )
        }
        configuration = @{ returnImmediately = $true }
    }
} | ConvertTo-Json

curl.exe -X POST http://localhost:8001/ `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer token_merchant_a" `
  -d $body
```

## Refund walkthrough (propose -> approve)

```powershell
# 1. Start a chat thread and request a refund
python client.py chat --token token_merchant_a --thread refund1 "Refund order 1042"
# Look for a [proposal] event with proposal_id

# 2. Approve the proposal
python client.py approve --token token_merchant_a --thread refund1 --proposal <proposal-id>

# 3. Verify the order is REFUNDED
curl.exe http://localhost:8002/orders/lookup?id=1042 `
  -H "Authorization: Bearer token_merchant_a"
```

## Reject walkthrough

```powershell
# 1. Propose a refund
python client.py chat --token token_merchant_a --thread refund2 "Refund order 1045"

# 2. Reject it
python client.py reject --token token_merchant_a --thread refund2 --proposal <proposal-id>

# 3. Verify order is still PAID
curl.exe http://localhost:8002/orders/lookup?id=1045 `
  -H "Authorization: Bearer token_merchant_a"
```

## Tenant isolation demo

Merchant A tries to access Merchant B's order (2001):

```powershell
# Lookup - should return "not found"
curl.exe http://localhost:8002/orders/lookup?id=2001 `
  -H "Authorization: Bearer token_merchant_a"

# Refund - should return 404 (identical to not found)
curl.exe -X POST http://localhost:8002/orders/refund `
  -H "Authorization: Bearer token_merchant_a" `
  -H "Content-Type: application/json" `
  -d '{"id":"2001"}'

# List - only merchant_a orders
curl.exe http://localhost:8002/orders `
  -H "Authorization: Bearer token_merchant_a"
```

## Restart persistence demo

With Docker Compose:

```powershell
# 1. Start a conversation and propose a refund
python client.py chat --token token_merchant_a --thread persist1 "Refund order 1042"
# Note the proposal_id from the [proposal] event

# 2. Restart the orchestrator
docker compose restart orchestrator

# 3. Continue the same thread - proposal persists
python client.py approve --token token_merchant_a --thread persist1 --proposal <proposal-id>

# 4. Verify the order is REFUNDED
curl.exe http://localhost:8002/orders/lookup?id=1042 `
  -H "Authorization: Bearer token_merchant_a"
```

## Requirements

- `requirements.txt` - runtime dependencies (pinned)
- `requirements-dev.txt` - test dependencies (`-r requirements.txt` + pytest, pytest-asyncio, respx)

## Project structure

```
store_api.py        R0 mock store API + seed data
orders_agent.py     R2 A2A server (deterministic, no LLM)
orchestrator.py     R1/R5/R6/R7 LangGraph graph + chat API (SSE, approve, reject)
guardrails.py       R4 redaction, injection neutralisation, fail-closed auth
llm.py              get_llm() factory + deterministic FakeChatModel
config.py           env settings only
evals.py            R9 eval harness: python evals.py
client.py           SSE demo client + approve/reject helpers
tests/              pytest suite
Dockerfile          multi-stage, non-root image
docker-compose.yml  4 services (store-api, orders-agent, orchestrator, postgres)
```
