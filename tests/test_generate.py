import asyncio
import io
import json
import logging

import pytest

from legal_rag.generate import (
    SYSTEM,
    GroqChat,
    StubChat,
    answer,
    answer_async,
    build_prompt,
    stream_answer,
)
from legal_rag.logging_conf import JsonFormatter, correlation_context
from legal_rag.rag import Hit

QUESTION = "ما هو سن الرشد؟"
ANSWER = "سن الرشد 21 سنة (المادة 44)"


class FakeChat:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return ANSWER


class FailingChat:
    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("groq is down")


def _hit(n: int, text: str) -> Hit:
    return Hit(article_number=n, citation=f"Egyptian Civil Code, Art. {n}", text_ar=text, score=0.9)


@pytest.fixture
def generate_log():
    """Capture the legal_rag.generate logger as parsed JSON lines."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("legal_rag.generate")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    yield lambda: [json.loads(line) for line in buf.getvalue().splitlines()]
    logger.removeHandler(handler)
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def test_build_prompt_includes_each_article_and_the_question():
    prompt = build_prompt(QUESTION, [_hit(44, "نص أ"), _hit(45, "نص ب")])
    assert "المادة 44: نص أ" in prompt
    assert "المادة 45: نص ب" in prompt
    assert prompt.endswith(f"السؤال: {QUESTION}")


def test_answer_uses_system_prompt_and_returns_model_text():
    chat = FakeChat()
    assert answer(QUESTION, [_hit(44, "نص")], chat=chat) == ANSWER
    assert chat.calls[0][0] == SYSTEM


def test_generation_log_carries_the_correlation_id_and_no_text(generate_log):
    with correlation_context("req-123"):
        answer(QUESTION, [_hit(44, "نص")], chat=FakeChat())
    (record,) = generate_log()
    assert record["correlation_id"] == "req-123"
    assert record["level"] == "INFO" and record["message"] == "answer generated"
    assert record["articles"] == [44] and record["latency_ms"] >= 0
    serialised = json.dumps(record, ensure_ascii=False)
    assert QUESTION not in serialised and ANSWER not in serialised


def test_generation_failure_is_an_error_with_correlation_id_and_is_reraised(generate_log):
    with correlation_context("req-456"), pytest.raises(RuntimeError, match="groq is down"):
        answer(QUESTION, [_hit(44, "نص")], chat=FailingChat())
    (record,) = generate_log()
    assert record["level"] == "ERROR" and record["correlation_id"] == "req-456"
    assert "RuntimeError: groq is down" in record["exception"]


def test_stub_chat_waits_the_configured_time_and_returns_text():
    import time

    start = time.perf_counter()
    text = StubChat(latency_ms=50).complete(SYSTEM, "x")
    assert time.perf_counter() - start >= 0.05
    assert text.strip()


def test_stub_chat_defaults_to_the_setting(monkeypatch):
    from legal_rag.config import settings

    monkeypatch.setattr(settings, "stub_chat_latency_ms", 0.0)
    assert StubChat()._delay_s == 0.0


# ---- async and streaming generation (the BentoML service) -----------------------------------


class AsyncFakeChat:
    def __init__(self, tokens=("نص ", "الإجابة")):
        self.tokens, self.calls = tokens, []

    async def acomplete(self, system, user):
        self.calls.append((system, user))
        return "".join(self.tokens)

    async def astream(self, system, user):
        self.calls.append((system, user))
        for t in self.tokens:
            yield t


def collect(agen):
    async def run():
        return [t async for t in agen]

    return asyncio.run(run())


def test_answer_async_returns_the_text_and_logs_metadata_only(generate_log):
    chat = AsyncFakeChat()
    with correlation_context("req-1"):
        text = asyncio.run(answer_async(QUESTION, [_hit(44, "نص")], chat))
    assert text == "نص الإجابة" and chat.calls[0][0] == SYSTEM
    (record,) = generate_log()
    assert record["message"] == "answer generated" and record["articles"] == [44]
    assert record["streamed"] is False and record["correlation_id"] == "req-1"
    assert QUESTION not in json.dumps(record, ensure_ascii=False)


def test_stream_answer_yields_tokens_in_order_and_logs_once(generate_log):
    tokens = collect(stream_answer(QUESTION, [_hit(44, "نص")], AsyncFakeChat()))
    assert tokens == ["نص ", "الإجابة"]
    (record,) = generate_log()
    assert record["streamed"] is True and record["answer_chars"] == len("نص الإجابة")


def test_stream_failure_is_logged_and_reraised(generate_log):
    class Broken:
        async def astream(self, system, user):
            yield "x"
            raise RuntimeError("stream cut")

    with pytest.raises(RuntimeError, match="stream cut"):
        collect(stream_answer(QUESTION, [_hit(44, "نص")], Broken()))
    assert generate_log()[0]["level"] == "ERROR"


def test_stub_chat_streams_the_same_text_it_completes():
    stub = StubChat(latency_ms=0)
    assert "".join(collect(stub.astream(SYSTEM, "x"))) == StubChat.TEXT
    assert asyncio.run(stub.acomplete(SYSTEM, "x")) == StubChat.TEXT


def test_stub_stream_delivers_the_first_token_before_the_end():
    import time

    start = time.perf_counter()
    arrivals = []

    async def run():
        async for _ in StubChat(latency_ms=200).astream(SYSTEM, "x"):
            arrivals.append(time.perf_counter() - start)

    asyncio.run(run())
    assert arrivals[0] < 0.12 and arrivals[-1] >= 0.19  # first word at ~30 %, last at ~100 %


class _Choice:
    def __init__(self, content=None, delta=None):
        from types import SimpleNamespace as NS

        self.message = NS(content=content)
        self.delta = NS(content=delta)


def test_groq_async_complete_and_stream(monkeypatch):
    from types import SimpleNamespace as NS

    from pydantic import SecretStr

    from legal_rag.config import settings

    monkeypatch.setattr(settings, "groq_api_key", SecretStr("k"))

    async def create(**kwargs):
        if kwargs.get("stream"):

            async def chunks():
                yield NS(choices=[])  # keep-alive chunk without choices
                for piece in ("ab", None, "", "cd"):
                    yield NS(choices=[_Choice(delta=piece)])

            return chunks()
        return NS(choices=[_Choice(content="كامل")])

    chat = GroqChat()
    chat._aclient = NS(chat=NS(completions=NS(create=create)))
    assert asyncio.run(chat.acomplete(SYSTEM, "x")) == "كامل"
    assert collect(chat.astream(SYSTEM, "x")) == ["ab", "cd"]
