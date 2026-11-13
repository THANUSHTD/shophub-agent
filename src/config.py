import os
from pathlib import Path

def _load_dotenv_if_present() -> None:

    candidates = [
        Path(".env"),
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for p in candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
            except Exception:
                pass


_load_dotenv_if_present()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


LLM_MODE = _env("LLM_MODE", "fake")
OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", _env("OPENAI_API_BASE", ""))

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
