"""Step 03: JSON log lines, correlation ids, log levels, and the no-print() rule."""

import ast
import asyncio
import io
import json
import logging
from pathlib import Path

import pytest

from legal_rag.logging_conf import (
    JsonFormatter,
    configure_logging,
    correlation_context,
    correlation_id_var,
)
from legal_rag.rag import IndexNotBuiltError, LegalRAG

SRC = Path(__file__).resolve().parents[1] / "src"
REQUIRED_KEYS = {"timestamp", "level", "logger", "message", "correlation_id"}


@pytest.fixture
def capture():
    """Attach a JSON handler to a named logger; returns a function reading parsed lines."""
    attached = []

    def _capture(name: str, level: int = logging.DEBUG):
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        attached.append((logger, handler))
        return lambda: [json.loads(line) for line in buf.getvalue().splitlines()]

    yield _capture
    for logger, handler in attached:
        logger.removeHandler(handler)
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def test_every_line_is_valid_json_with_the_required_keys(capture):
    lines = capture("legal_rag.demo")
    logging.getLogger("legal_rag.demo").info("hello")
    (record,) = lines()
    assert REQUIRED_KEYS <= record.keys()
    assert record["level"] == "INFO" and record["message"] == "hello"


def test_correlation_id_is_set_inside_the_context_and_reset_after(capture):
    lines = capture("legal_rag.demo")
    log = logging.getLogger("legal_rag.demo")
    log.info("before")
    with correlation_context("req-1"):
        log.info("inside")
    log.info("after")
    assert [r["correlation_id"] for r in lines()] == ["-", "req-1", "-"]
    assert correlation_id_var.get() == "-"


def test_context_generates_a_uuid_when_none_is_given():
    with correlation_context() as cid:
        assert len(cid) == 32
        assert correlation_id_var.get() == cid


def test_concurrent_tasks_never_see_each_others_correlation_id(capture):
    lines = capture("legal_rag.demo")
    log = logging.getLogger("legal_rag.demo")

    async def handle(cid: str, delay: float) -> None:
        with correlation_context(cid):
            await asyncio.sleep(delay)
            log.info("working", extra={"who": cid})

    async def main() -> None:
        await asyncio.gather(handle("A", 0.02), handle("B", 0.01))

    asyncio.run(main())
    assert all(r["correlation_id"] == r["who"] for r in lines())


def test_extra_fields_and_tracebacks_are_serialised(capture):
    lines = capture("legal_rag.demo")
    log = logging.getLogger("legal_rag.demo")
    log.info("with extras", extra={"latency_ms": 12.5, "articles": [44, 46]})
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("failed")
    first, second = lines()
    assert first["latency_ms"] == 12.5 and first["articles"] == [44, 46]
    assert second["level"] == "ERROR" and "RuntimeError: boom" in second["exception"]


def test_arabic_text_is_not_escaped(capture):
    lines = capture("legal_rag.demo")
    logging.getLogger("legal_rag.demo").info("سن الرشد")
    assert lines()[0]["message"] == "سن الرشد"


def test_configure_logging_keeps_noisy_libraries_quiet():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("INFO")
        assert logging.getLogger("httpx").level == logging.WARNING
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        root.handlers[:], root.level = saved_handlers, saved_level


def test_no_print_calls_in_src():
    offenders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            is_print = isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print"
            if is_print:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


# ---- log levels in the RAG code -------------------------------------------------------


def test_served_question_is_logged_at_info_with_latency(rag, capture):
    lines = capture("legal_rag.rag", level=logging.INFO)
    rag.retrieve("ما هو سن الرشد", k=2)
    served = next(r for r in lines() if r["message"] == "question served")
    assert served["level"] == "INFO" and served["k"] == 2
    assert served["top_article"] == 44 and served["latency_ms"] >= 0


def test_question_text_only_appears_at_debug(rag, capture):
    info_lines = capture("legal_rag.rag", level=logging.INFO)
    rag.retrieve("ما هو سن الرشد")
    assert "سن الرشد" not in json.dumps(info_lines(), ensure_ascii=False)
    debug_lines = capture("legal_rag.rag", level=logging.DEBUG)
    rag.retrieve("ما هو سن الرشد")
    assert any(r.get("question") == "ما هو سن الرشد" for r in debug_lines())


def test_low_confidence_is_a_warning(articles, embedder, index_dir, capture):
    lines = capture("legal_rag.rag", level=logging.INFO)
    strict = LegalRAG(embedder, index_dir, articles, low_score_threshold=0.99)
    strict.retrieve("وصفة كشري")
    assert [r["level"] for r in lines() if r["message"] == "low retrieval confidence"] == [
        "WARNING"
    ]
    relaxed = LegalRAG(embedder, index_dir, articles, low_score_threshold=0.0)
    relaxed.retrieve("وصفة كشري")
    assert len([r for r in lines() if r["message"] == "low retrieval confidence"]) == 1


def test_rejected_question_is_a_warning_not_an_error(rag, capture):
    lines = capture("legal_rag.rag", level=logging.INFO)
    with pytest.raises(ValueError):
        rag.retrieve("  ")
    (record,) = lines()
    assert record["level"] == "WARNING"


def test_missing_index_is_an_error_with_a_clear_message(embedder, articles, tmp_path, capture):
    lines = capture("legal_rag.rag", level=logging.INFO)
    with pytest.raises(IndexNotBuiltError, match="ingest"):
        LegalRAG(embedder, tmp_path / "empty", articles)
    assert [r["level"] for r in lines()] == ["ERROR"]
