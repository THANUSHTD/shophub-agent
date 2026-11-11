"""Application configuration. Reads environment variables only (fail-closed).

Never log or return the values below. Defaults are safe for local development
and Docker Compose; production must override via environment.
"""

import os


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


LLM_MODE = _env("LLM_MODE", "fake")
OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4.1-mini")

CHECKPOINTER = _env("CHECKPOINTER", "memory")
DATABASE_URL = _env("DATABASE_URL", "postgresql://shophub:shophub@postgres:5432/shophub")

STORE_API_URL = _env("STORE_API_URL", "http://store-api:8002")
ORDERS_AGENT_URL = _env("ORDERS_AGENT_URL", "http://orders-agent:8001")
ORDERS_AGENT_PUBLIC_URL = _env("ORDERS_AGENT_PUBLIC_URL", "http://orders-agent:8001")

ORCHESTRATOR_PORT = int(_env("ORCHESTRATOR_PORT", "8000"))
ORDERS_AGENT_PORT = int(_env("ORDERS_AGENT_PORT", "8001"))
STORE_API_PORT = int(_env("STORE_API_PORT", "8002"))

# Tokens are MOCK FIXTURES only - the only credentials allowed in this repo.
MERCHANT_TOKENS = {
    "token_merchant_a": {"user": "support_a", "merchant": "merchant_a"},
    "token_merchant_b": {"user": "support_b", "merchant": "merchant_b"},
    "token_no_merchant": {"user": "support_none", "merchant": None},
}
