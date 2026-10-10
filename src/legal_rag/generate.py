"""Answer generation: retrieved articles -> Groq LLM -> Arabic answer."""

import logging
import time
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


class GroqChat:
    def __init__(self, model: str | None = None) -> None:
        from groq import Groq  # lazy import

        key = settings.groq_api_key
        if key is None:
            raise RuntimeError("GROQ_API_KEY is not set: put it in .env")
        self._client = Groq(api_key=key.get_secret_value())
        self._model = model or settings.groq_model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            max_completion_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"المادة {h.article_number}: {h.text_ar}" for h in hits)
    return f"المواد:\n{context}\n\nالسؤال: {question}"


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
    log.info(
        "answer generated",
        extra={
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "answer_chars": len(text),
            "articles": articles,
        },
    )
    return text
