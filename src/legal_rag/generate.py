"""Answer generation: retrieved articles -> Groq LLM -> Arabic answer."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Protocol

from legal_rag.config import settings
from legal_rag.rag import Hit

log = logging.getLogger(__name__)

SYSTEM = (
    "أنت مساعد قانوني. أجب بالعربية الفصحى وبالاعتماد فقط على المواد المرفقة. "
    "اذكر رقم المادة التي اعتمدت عليها. إذا لم تحتوِ المواد على الإجابة فقل ذلك صراحة."
)
MAX_TOKENS = 400


class ChatClient(Protocol):
    """The slice of the Groq client we use; tests pass a fake."""

    def complete(self, system: str, user: str) -> str: ...


class AsyncChatClient(Protocol):
    """What the async serving layer needs: a whole answer, or the answer as a token stream."""

    async def acomplete(self, system: str, user: str) -> str: ...

    def astream(self, system: str, user: str) -> AsyncIterator[str]: ...


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class GroqChat:
    def __init__(self, model: str | None = None) -> None:
        from groq import Groq  # lazy import

        key = settings.groq_api_key
        if key is None:
            raise RuntimeError("GROQ_API_KEY is not set: put it in .env")
        self._key = key.get_secret_value()
        self._client = Groq(api_key=self._key)
        self._aclient = None  # built on first async use, inside the running event loop
        self._model = model or settings.groq_model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            max_completion_tokens=MAX_TOKENS,
            messages=_messages(system, user),
        )
        return resp.choices[0].message.content or ""

    def _async_client(self):
        if self._aclient is None:
            from groq import AsyncGroq

            self._aclient = AsyncGroq(api_key=self._key)
        return self._aclient

    async def acomplete(self, system: str, user: str) -> str:
        resp = await self._async_client().chat.completions.create(
            model=self._model,
            temperature=0,
            max_completion_tokens=MAX_TOKENS,
            messages=_messages(system, user),
        )
        return resp.choices[0].message.content or ""

    async def astream(self, system: str, user: str) -> AsyncIterator[str]:
        stream = await self._async_client().chat.completions.create(
            model=self._model,
            temperature=0,
            max_completion_tokens=MAX_TOKENS,
            messages=_messages(system, user),
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


class StubChat:
    """A language model that answers after a fixed delay: for load tests only.

    The delay is a plain sleep, like the wait for a hosted LLM's reply (the thread is idle and
    the GIL is free), so the stub keeps the shape of the real call while taking the provider's
    rate limits, network and cost out of the measurement.
    """

    def __init__(self, latency_ms: float | None = None) -> None:
        ms = settings.stub_chat_latency_ms if latency_ms is None else latency_ms
        self._delay_s = max(ms, 0.0) / 1000

    TEXT = "إجابة تجريبية من نموذج وهمي، لا تُستخدم إلا في اختبار الحمل."

    def complete(self, system: str, user: str) -> str:
        time.sleep(self._delay_s)
        return self.TEXT

    async def acomplete(self, system: str, user: str) -> str:
        await asyncio.sleep(self._delay_s)
        return self.TEXT

    async def astream(self, system: str, user: str) -> AsyncIterator[str]:
        """The first word after 30 % of the latency, the rest spread over the remaining 70 %."""
        words = self.TEXT.split(" ")
        await asyncio.sleep(self._delay_s * 0.3)
        for i, word in enumerate(words):
            if i:
                await asyncio.sleep(self._delay_s * 0.7 / (len(words) - 1))
            yield word if i == len(words) - 1 else word + " "


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"المادة {h.article_number}: {h.text_ar}" for h in hits)
    return f"المواد:\n{context}\n\nالسؤال: {question}"


def _log_generated(start: float, chars: int, articles: list[int], streamed: bool = False) -> None:
    log.info(
        "answer generated",
        extra={
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "answer_chars": chars,
            "articles": articles,
            "streamed": streamed,
        },
    )


def answer(question: str, hits: list[Hit], chat: ChatClient | None = None) -> str:
    """Generate the answer. Never logs the question or answer text (privacy), only metadata."""
    articles = [h.article_number for h in hits]
    start = time.perf_counter()
    try:
        chat = chat or GroqChat()
        text = chat.complete(SYSTEM, build_prompt(question, hits))
    except Exception:
        log.exception("generation failed", extra={"articles": articles})
        raise
    _log_generated(start, len(text), articles)
    return text


async def answer_async(question: str, hits: list[Hit], chat: AsyncChatClient) -> str:
    """`answer` without blocking the event loop: the wait for the LLM is an `await`."""
    articles = [h.article_number for h in hits]
    start = time.perf_counter()
    try:
        text = await chat.acomplete(SYSTEM, build_prompt(question, hits))
    except Exception:
        log.exception("generation failed", extra={"articles": articles})
        raise
    _log_generated(start, len(text), articles)
    return text


async def stream_answer(
    question: str, hits: list[Hit], chat: AsyncChatClient
) -> AsyncIterator[str]:
    """The answer token by token. Logged once at the end, with metadata only."""
    articles = [h.article_number for h in hits]
    start = time.perf_counter()
    chars = 0
    try:
        async for token in chat.astream(SYSTEM, build_prompt(question, hits)):
            chars += len(token)
            yield token
    except Exception:
        log.exception("generation failed", extra={"articles": articles, "streamed": True})
        raise
    _log_generated(start, chars, articles, streamed=True)
