"""/model says which retriever answers /ask: a registry version or the baked-in Chroma index."""

import numpy as np
from fastapi.testclient import TestClient

from legal_rag.api.main import create_app
from legal_rag.config import settings
from legal_rag.data import Article
from legal_rag.registry_rag import Bundle, RegistryRAG

X = np.array([1.0, 0.0], dtype=np.float32)


class FakeChat:
    def complete(self, system: str, user: str) -> str:
        return "answer (Article 10)"


class FixedEmbedder:
    def embed_query(self, text):
        return list(X)

    def embed_passages(self, texts):
        raise NotImplementedError


def registry_rag() -> RegistryRAG:
    article = Article(
        article_number=10, text_ar="نص", text_en="", is_repealed=False,
        citation="Art. 10", book="", chapter="", section="", topic="", source_page=1,
    )  # fmt: skip
    config = {
        "embedding_model": settings.embedding_model,
        "variants": ["ar", "en"],
        "fusion": "max",
        "weights": [],
        "corpus_sha256": "abc",
    }
    bundle = Bundle(config, [{"article_number": 10, "citation": "Art. 10"}], [X[None], X[None]])
    info = {"name": "legal-rag-retriever", "version": "2", "alias": "production", "run_id": "r1"}
    info |= {"embedding_model": config["embedding_model"]}
    info |= {"variants": ["ar", "en"], "fusion": "max"}
    return RegistryRAG(bundle, FixedEmbedder(), articles=[article], info=info)


def test_model_reports_the_registry_version():
    with TestClient(create_app(rag=registry_rag(), chat=FakeChat())) as c:
        body = c.get("/model").json()
    assert body["source"] == "registry" and body["version"] == "2"
    assert body["alias"] == "production" and body["fusion"] == "max"


def test_model_reports_the_baked_in_index_for_chroma(rag):
    with TestClient(create_app(rag=rag, chat=FakeChat())) as c:
        body = c.get("/model").json()
    assert body["source"] == "chroma" and body["variants"] == ["ar"]
    assert body["version"] is None


def test_ask_is_served_by_the_registry_retriever():
    with TestClient(create_app(rag=registry_rag(), chat=FakeChat())) as c:
        r = c.post("/ask", json={"question": "ما هو؟", "k": 1})
    assert r.status_code == 200 and r.json()["sources"] == ["Art. 10"]


def test_model_is_503_when_nothing_is_loaded():
    r = TestClient(create_app(rag=registry_rag(), chat=FakeChat())).get("/model")
    assert r.status_code == 503
