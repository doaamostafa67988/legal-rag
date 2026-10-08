"""API tests: real LegalRAG on the tiny fake-embedder index (conftest), fake LLM, no network."""

import hashlib
import io
import json
import logging
import re
import warnings

import pytest
from fastapi.testclient import TestClient

from legal_rag.api import main as api_main
from legal_rag.api.main import create_app
from legal_rag.config import settings
from legal_rag.logging_conf import JsonFormatter

ANSWER = "سن الرشد 21 سنة (المادة 44)"
QUESTION = "ما هو سن الرشد"


class FakeChat:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return ANSWER


class FailingChat:
    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("secret internal detail")


@pytest.fixture(autouse=True)
def _isolate_global_logging():
    """The app's lifespan reconfigures root logging; undo it so other tests are unaffected."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    with warnings.catch_warnings():
        yield
    root.handlers[:], root.level = handlers, level
    logging.captureWarnings(False)


@pytest.fixture
def api_log():
    """Capture every legal_rag.* log line as parsed JSON."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("legal_rag")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    yield lambda: [json.loads(line) for line in buf.getvalue().splitlines()]
    logger.removeHandler(handler)
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


@pytest.fixture
def chat() -> FakeChat:
    return FakeChat()


@pytest.fixture
def client(rag, chat):
    with TestClient(create_app(rag=rag, chat=chat)) as c:  # `with` runs the lifespan
        yield c


# ---- /health -------------------------------------------------------------------------


def test_health_reports_the_index_size(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "documents_indexed": 3, "embedder": "FakeEmbedder"}


def test_health_is_503_when_the_model_is_not_loaded(rag, chat):
    # no `with`: the lifespan never runs, so the process is alive but nothing is loaded
    r = TestClient(create_app(rag=rag, chat=chat)).get("/health")
    assert r.status_code == 503


# ---- /metadata -----------------------------------------------------------------------


def test_metadata_describes_what_is_being_served(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "service_version",
        "embedding_model",
        "embedder",
        "documents_indexed",
        "groq_model",
        "corpus_sha256",
    }
    assert body["documents_indexed"] == 3
    assert body["embedder"] == "FakeEmbedder"
    assert body["embedding_model"] == settings.embedding_model
    assert body["groq_model"] == settings.groq_model
    assert body["service_version"]


def test_metadata_hash_matches_the_corpus_file(monkeypatch, rag, chat, tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_bytes(b"[]")
    monkeypatch.setattr(api_main.settings, "corpus_json", corpus)
    with TestClient(create_app(rag=rag, chat=chat)) as c:
        assert c.get("/metadata").json()["corpus_sha256"] == hashlib.sha256(b"[]").hexdigest()


def test_metadata_hash_is_null_when_the_corpus_file_is_missing(monkeypatch, rag, chat, tmp_path):
    monkeypatch.setattr(api_main.settings, "corpus_json", tmp_path / "missing.json")
    with TestClient(create_app(rag=rag, chat=chat)) as c:
        assert c.get("/metadata").json()["corpus_sha256"] is None


def test_metadata_is_503_when_the_model_is_not_loaded(rag, chat):
    assert TestClient(create_app(rag=rag, chat=chat)).get("/metadata").status_code == 503


# ---- /ask ----------------------------------------------------------------------------


def test_ask_returns_answer_and_article_citations(client):
    r = client.post("/ask", json={"question": QUESTION})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "sources", "correlation_id", "latency_ms"}
    assert body["answer"] == ANSWER
    assert body["sources"][0] == "Egyptian Civil Code, Article 44"
    assert all(s.startswith("Egyptian Civil Code, Article ") for s in body["sources"])


def test_ask_respects_k(client):
    assert len(client.post("/ask", json={"question": QUESTION, "k": 1}).json()["sources"]) == 1


def test_docs_page_and_example_are_served(client):
    assert client.get("/docs").status_code == 200
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert schemas["AskRequest"]["example"]["question"]


# ---- validation -> 422 ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"question": "   "}, "question"),
        ({"question": ""}, "question"),
        ({}, "question"),
        ({"question": "x" * 1001}, "question"),
        ({"question": "x", "k": 0}, "k"),
        ({"question": "x", "k": 11}, "k"),
    ],
)
def test_invalid_payload_is_a_readable_422(client, chat, payload, field):
    r = client.post("/ask", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"][0]["field"] == field
    assert "Traceback" not in r.text
    assert chat.calls == 0  # rejected before any LLM call


# ---- correlation id ------------------------------------------------------------------


def test_header_and_body_carry_the_same_correlation_id(client):
    r = client.post("/ask", json={"question": QUESTION})
    assert r.headers["x-request-id"] == r.json()["correlation_id"]


def test_a_client_supplied_request_id_is_echoed(client):
    r = client.post("/ask", json={"question": QUESTION}, headers={"X-Request-ID": "my-id-1"})
    assert r.headers["x-request-id"] == r.json()["correlation_id"] == "my-id-1"


def test_an_unsafe_request_id_is_replaced(client):
    r = client.get("/health", headers={"X-Request-ID": "bad id with spaces!"})
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["x-request-id"])


def test_one_request_one_correlation_id_across_every_log_line(client, api_log):
    r = client.post("/ask", json={"question": QUESTION})
    cid = r.headers["x-request-id"]
    lines = {rec["message"]: rec for rec in api_log() if rec["correlation_id"] == cid}
    assert {"question served", "answer generated", "request"} <= lines.keys()
    assert lines["request"]["status"] == 200 and lines["request"]["path"] == "/ask"


def test_question_and_answer_text_are_not_logged_at_info(client, api_log):
    client.post("/ask", json={"question": QUESTION})
    logged = json.dumps(api_log(), ensure_ascii=False)
    assert QUESTION not in logged and ANSWER not in logged


# ---- unexpected errors -> clean 500 --------------------------------------------------


def test_unexpected_error_is_a_clean_500_that_is_logged_with_its_id(rag, api_log):
    app = create_app(rag=rag, chat=FailingChat())
    with TestClient(app) as c:
        r = c.post("/ask", json={"question": QUESTION})
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error"
    assert "secret internal detail" not in r.text and "Traceback" not in r.text
    cid = r.headers["x-request-id"]
    errors = [rec for rec in api_log() if rec["level"] == "ERROR" and rec["correlation_id"] == cid]
    assert any("secret internal detail" in rec["exception"] for rec in errors)


# ---- startup (monkeypatch: no real model or Groq key needed) -------------------------


def test_startup_loads_the_model_once(monkeypatch, rag):
    loads = []
    monkeypatch.setattr(api_main, "_load_rag", lambda: loads.append(1) or rag)
    monkeypatch.setattr(api_main, "GroqChat", FakeChat)
    with TestClient(api_main.create_app()) as c:
        for _ in range(3):
            assert c.post("/ask", json={"question": QUESTION}).status_code == 200
    assert loads == [1]


def test_startup_failure_is_logged_as_an_error(monkeypatch, api_log):
    def boom():
        raise RuntimeError("index missing")

    monkeypatch.setattr(api_main, "_load_rag", boom)
    with pytest.raises(RuntimeError, match="index missing"), TestClient(api_main.create_app()):
        pass
    levels = [r["level"] for r in api_log() if r["message"].startswith("startup failed")]
    assert levels == ["ERROR"]
