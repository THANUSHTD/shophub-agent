"""LLM factory + deterministic FakeChatModel.

get_llm() reads LLM_MODE (fake|openai). OpenAI mode uses ChatOpenAI with the
key from OPENAI_API_KEY. Fake mode is a scripted BaseChatModel that decides
tool calls from the user's message text and records every prompt in seen_prompts
so tests can assert no PII reached the model.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Iterator, AsyncIterator, Sequence, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from pydantic import PrivateAttr
from langchain_openai import ChatOpenAI

from config import LLM_MODE, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL


# Order ids are exactly 4 digits.
ORDER_ID_RE = re.compile(r"\b(\d{4})\b")
# A customer name: one or more capitalised words.
NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

_REFUND_RE = re.compile(r"\brefund\s+order\s+(\d{4})\b", re.IGNORECASE)
_CANCEL_RE = re.compile(r"\bcancel\s+order\s+(\d{4})\b", re.IGNORECASE)
_LOOKUP_RE = re.compile(
    r"\b(?:find|show|lookup|get)\s+order\s+(\d{4})\b", re.IGNORECASE
)
_CUSTOMER_RE = re.compile(
    r"\b(?:what did|customer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", re.IGNORECASE
)
_STATUS_ORDERS_RE = re.compile(
    r"\b(placed|shipped|delivered|cancelled|refunded)\s+orders\b", re.IGNORECASE
)
_PAID_ORDERS_RE = re.compile(r"\bpaid\s+orders\b", re.IGNORECASE)
_OLDER_PAID_RE = re.compile(
    r"\bpaid\s+orders\s+older\s+than\s+(\d+)\s+days\b", re.IGNORECASE
)
_NOT_SHIPPED_RE = re.compile(
    r"\bnot\s+shipped\s+(?:in|for)\s+(\d+)\s+days\b", re.IGNORECASE
)

HELP = (
    "I can help with your store's orders. Try: 'show paid orders', "
    "'paid orders older than 3 days', 'find order 1042', 'what did Maria "
    "Fernandes order', 'refund order 1042' or 'cancel order 1044'. "
    "Refunds and cancellations are proposed for your approval first."
)


class FakeChatModel(BaseChatModel):
    """Deterministic scripted chat model.

    Decides tool calls from the latest human message via regex. Streams real
    chunks. Records every prompt it receives in `seen_prompts`. `obedient=True`
    (tests/evals only) also emits propose_refund when it sees "refund" inside
    tool output text.
    """

    obedient: bool = False
    seen_prompts: list[str] = []
    _bound_tools: list[Any] = PrivateAttr(default_factory=list)

    def __init__(self, obedient: bool = False, **kwargs: Any):
        super().__init__(obedient=obedient, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake-shophub"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": "fake-shophub", "obedient": self.obedient}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[Any] | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        self._bound_tools = list(tools or [])
        return self

    # -- sync stubs (we stream via _astream) --------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        chunks = list(self._stream(messages, stop, run_manager, **kwargs))
        gens = [
            ChatGeneration(message=chunk.message) for chunk in chunks
        ]
        return ChatResult(generations=gens)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self.seen_prompts.append(self._prompt_text(messages))
        yield from self._emit(messages)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.seen_prompts.append(self._prompt_text(messages))
        for chunk in self._emit(messages):
            yield chunk

    # -- decision / emission ----------------------------------------------

    def _prompt_text(self, messages: list[BaseMessage]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m.type if hasattr(m, "type") else m.__class__.__name__
            content = getattr(m, "content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _last_human_text(self, messages: list[BaseMessage]) -> str:
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                return getattr(m, "content", "") or ""
        return ""

    def _last_tool_text(self, messages: list[BaseMessage]) -> str:
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                return getattr(m, "content", "") or ""
        return ""

    def _decide_tool(self, last_human: str, messages: list[BaseMessage]):
        text = last_human.strip()
        m = _REFUND_RE.search(text)
        if m:
            return "propose_refund", {"id": m.group(1)}
        m = _CANCEL_RE.search(text)
        if m:
            return "propose_cancel", {"id": m.group(1)}
        m = _OLDER_PAID_RE.search(text)
        if m:
            return "list_orders", {"status": "PAID", "older_than_days": int(m.group(1))}
        m = _NOT_SHIPPED_RE.search(text)
        if m:
            return "list_orders", {"status": "PAID", "older_than_days": int(m.group(1))}
        m = _PAID_ORDERS_RE.search(text)
        if m:
            return "list_orders", {"status": "PAID"}
        m = _STATUS_ORDERS_RE.search(text)
        if m:
            return "list_orders", {"status": m.group(1).upper()}
        m = _LOOKUP_RE.search(text)
        if m:
            return "lookup_order", {"id": m.group(1)}
        m = _CUSTOMER_RE.search(text)
        if m:
            return "search_customer_orders", {"name": m.group(1)}
        return None

    def _emit(self, messages: list[BaseMessage]) -> Iterator[ChatGenerationChunk]:
        last_human = self._last_human_text(messages)
        # If the previous turn was a tool result, compose a final answer instead
        # of re-deciding a tool call (except obedient-mode refund injection).
        last_is_tool = any(isinstance(m, ToolMessage) for m in messages[-2:])
        if last_is_tool:
            tool_text = self._last_tool_text(messages)
            if self.obedient and "refund" in tool_text.lower():
                m = ORDER_ID_RE.search(tool_text)
                if m:
                    yield self._tool_chunk("propose_refund", {"id": m.group(1)})
                    return
            yield from self._answer_chunks(self._compose_answer(messages))
            return

        decision = self._decide_tool(last_human, messages)
        if decision is None:
            yield from self._answer_chunks(HELP)
            return
        name, args = decision
        yield self._tool_chunk(name, args)

    def _tool_chunk(self, name: str, args: dict[str, Any]) -> ChatGenerationChunk:
        tc = {"id": "call_fake_1", "name": name, "args": args}
        return ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=[tc]))

    def _answer_chunks(self, text: str) -> Iterator[ChatGenerationChunk]:
        for ch in text:
            yield ChatGenerationChunk(message=AIMessageChunk(content=ch))

    def _compose_answer(self, messages: list[BaseMessage]) -> str:
        """Build a readable final answer from the tool JSON in the history."""
        # Gather tool results in reverse (latest first).
        tool_results: list[tuple[str, str]] = []
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                tool_results.append((getattr(m, "name", "tool"), getattr(m, "content", "") or ""))
                if len(tool_results) >= 3:
                    break
        if not tool_results:
            return HELP
        # Use the latest tool result primarily.
        name, content = tool_results[0]
        answer = self._format_tool_result(name, content)
        if len(tool_results) > 1:
            answer += "\n\n" + self._format_tool_result(tool_results[1][0], tool_results[1][1])
        return answer

    def _format_tool_result(self, name: str, content: str) -> str:
        import json

        try:
            data = json.loads(content)
        except Exception:
            return content

        # Strip the untrusted prefix if present.
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            return content

        if not data.get("ok"):
            error = data.get("error", "unknown")
            message = data.get("message", "")
            if error == "ORDER_NOT_FOUND":
                return f"Order not found. {message}".strip()
            if error == "NO_MERCHANT":
                return "You are not scoped to a merchant."
            if error == "INVALID_STATE":
                return f"That action isn't allowed for this order's current status. {message}".strip()
            return f"Request failed ({error}). {message}".strip()

        payload = data.get("data")
        if name == "list_orders":
            return self._format_list(payload)
        if name == "lookup_order":
            return self._format_order(payload)
        if name == "search_customer_orders":
            return self._format_list(payload)
        if name in ("refund_order", "cancel_order", "propose_refund", "propose_cancel"):
            return self._format_order(payload) if isinstance(payload, dict) else str(payload)
        return json.dumps(payload)[:500]

    @staticmethod
    def _format_order(o: Any) -> str:
        if not isinstance(o, dict):
            return str(o)
        return (
            f"Order {o.get('id')}: {o.get('customer_name')} - status {o.get('status')} "
            f"- total {o.get('total')} (placed {o.get('placed_at')}). "
            f"{'Customer note: ' + o.get('customer_note') if o.get('customer_note') else ''}"
        ).strip()

    @staticmethod
    def _format_list(orders: Any) -> str:
        if not isinstance(orders, list):
            return str(orders)
        if not orders:
            return "No orders found."
        lines = ["Orders:"]
        for o in orders[:20]:
            if isinstance(o, dict):
                flag = " [note redacted/flagged]" if o.get("note_flagged") else ""
                lines.append(
                    f"- {o.get('id')}: {o.get('customer_name')} - {o.get('status')} - {o.get('total')}{flag}"
                )
        return "\n".join(lines)


def get_llm():
    """Factory: returns a ChatOpenAI (openai mode) or FakeChatModel (fake mode)."""
    mode = os.environ.get("LLM_MODE", LLM_MODE)
    if mode == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when LLM_MODE=openai. Set it in the environment or .env."
            )
        model = os.environ.get("OPENAI_MODEL", OPENAI_MODEL)
        base_url = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", OPENAI_BASE_URL))
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "streaming": True,
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    return FakeChatModel(obedient=False)
