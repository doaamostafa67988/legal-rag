import io
import json
import logging

import pytest

from legal_rag.generate import SYSTEM, answer, build_prompt
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
