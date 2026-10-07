"""Parity: the ONNX encoder must reproduce the PyTorch embeddings (Module 1, Step 4)."""

import numpy as np
import pytest

from legal_rag.config import settings
from legal_rag.data import load_corpus
from legal_rag.onnx_embed import OnnxEmbedder
from legal_rag.rag import E5Embedder

pytestmark = pytest.mark.integration  # needs the real model; run with -m integration

N = 500


@pytest.fixture(scope="module")
def texts() -> list[str]:
    articles = [a for a in load_corpus() if not a.is_repealed and a.text_ar.strip()]
    return [a.text_ar for a in articles[:N]]


@pytest.fixture(scope="module")
def onnx_embedder() -> OnnxEmbedder:
    if not (settings.onnx_dir / "model.onnx").exists():
        pytest.skip("run `python -m legal_rag.export` first")
    return OnnxEmbedder()


def test_onnx_matches_torch(texts, onnx_embedder):
    ref = np.array(E5Embedder().embed_passages(texts))
    got = np.array(onnx_embedder.embed_passages(texts))
    assert ref.shape == got.shape == (len(texts), 384)
    assert np.allclose(ref, got, atol=1e-4)
    assert (ref * got).sum(axis=1).min() > 0.9999  # cosine similarity per article


def test_dynamic_batch_axis(texts, onnx_embedder):
    batched = np.array(onnx_embedder.embed_passages(texts[:7], batch_size=7))
    single = np.array([onnx_embedder.embed_passages([t])[0] for t in texts[:7]])
    assert np.allclose(batched, single, atol=1e-4)
