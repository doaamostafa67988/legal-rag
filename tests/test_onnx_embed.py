"""OnnxEmbedder pooling/normalisation logic, with a fake tokenizer and session (no model)."""

import numpy as np
import pytest

from legal_rag.onnx_embed import OnnxEmbedder


class FakeTokenizer:
    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def __call__(self, texts, **kwargs):
        self.seen.append(list(texts))
        n = len(texts)
        return {
            "input_ids": np.ones((n, 3), dtype=np.int64),
            "attention_mask": np.array([[1, 1, 0]] * n, dtype=np.int64),  # last token is padding
        }


class FakeSession:
    def run(self, _outputs, feed):
        n = feed["input_ids"].shape[0]
        # two real tokens (3, 4) and one padding token (100, 100) that must be ignored
        return [np.array([[[3.0, 4.0], [3.0, 4.0], [100.0, 100.0]]] * n, dtype=np.float32)]


@pytest.fixture
def embedder() -> OnnxEmbedder:
    emb = object.__new__(OnnxEmbedder)  # skip __init__: no onnxruntime session, no files
    emb._tokenizer = FakeTokenizer()
    emb._session = FakeSession()
    return emb


def test_mean_pooling_ignores_padding_and_output_is_unit_length(embedder):
    assert embedder.embed_query("x") == pytest.approx([0.6, 0.8])


def test_query_and_passage_prefixes_and_batching(embedder):
    out = embedder.embed_passages(["a", "b", "c", "d", "e"], batch_size=2)
    assert len(out) == 5
    assert embedder._tokenizer.seen == [
        ["passage: a", "passage: b"],
        ["passage: c", "passage: d"],
        ["passage: e"],
    ]
    embedder.embed_query("q")
    assert embedder._tokenizer.seen[-1] == ["query: q"]


def test_init_loads_tokenizer_and_session_from_the_export_dir(monkeypatch, tmp_path):
    created = {}

    def fake_session(path, providers):
        created.update(path=path, providers=providers)
        return FakeSession()

    monkeypatch.setattr("onnxruntime.InferenceSession", fake_session)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda directory: FakeTokenizer()
    )
    emb = OnnxEmbedder(tmp_path)
    assert created["path"] == str(tmp_path / "model.onnx")
    assert created["providers"] == ["CPUExecutionProvider"]
    assert emb.embed_query("x") == pytest.approx([0.6, 0.8])
