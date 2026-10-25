"""evaluate.load_rag picks the retriever the way the API does (registry URI or baked-in index)."""

import legal_rag.evaluate as ev
import legal_rag.onnx_embed as onnx_embed
import legal_rag.rag as rag
import legal_rag.registry_rag as registry_rag
from legal_rag.config import settings

URI = "models:/legal-rag-retriever@production"


class FakeOnnx:
    pass


class FakeE5:
    pass


def test_registry_uri_set_serves_the_registered_retriever(monkeypatch):
    seen = {}
    monkeypatch.setattr(settings, "registry_model_uri", URI)
    monkeypatch.setattr(settings, "embedder_backend", "onnx")
    monkeypatch.setattr(onnx_embed, "OnnxEmbedder", FakeOnnx)

    def fake_load(uri, embedder):
        seen.update(uri=uri, embedder=embedder)
        return "registry retriever"

    monkeypatch.setattr(registry_rag, "load_from_registry", fake_load)
    assert ev.load_rag() == "registry retriever"
    assert seen["uri"] == URI and isinstance(seen["embedder"], FakeOnnx)


def test_registry_with_the_torch_backend_falls_back_to_the_e5_embedder(monkeypatch):
    seen = {}
    monkeypatch.setattr(settings, "registry_model_uri", URI)
    monkeypatch.setattr(rag, "E5Embedder", FakeE5)
    monkeypatch.setattr(
        registry_rag, "load_from_registry", lambda uri, embedder: seen.update(embedder=embedder)
    )
    ev.load_rag("torch")
    assert isinstance(seen["embedder"], FakeE5)


def test_without_a_registry_uri_the_baked_in_index_is_used(monkeypatch):
    built = {}

    class FakeRag:
        def __init__(self, embedder=None):
            built["embedder"] = embedder

    monkeypatch.setattr(settings, "registry_model_uri", None)
    monkeypatch.setattr(onnx_embed, "OnnxEmbedder", FakeOnnx)
    monkeypatch.setattr(ev, "LegalRAG", FakeRag)
    ev.load_rag("onnx")
    assert isinstance(built["embedder"], FakeOnnx)
    ev.load_rag("torch")
    assert built["embedder"] is None  # torch backend: LegalRAG builds its own E5Embedder
