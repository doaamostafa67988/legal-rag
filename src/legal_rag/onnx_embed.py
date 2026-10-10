"""Embedder backed by the exported ONNX encoder (same interface as E5Embedder)."""

from pathlib import Path

import numpy as np

from legal_rag.config import settings


class OnnxEmbedder:
    """multilingual-e5 on onnxruntime; pooling and normalisation happen here, not in the graph."""

    def __init__(self, onnx_dir: Path | None = None) -> None:
        import onnxruntime as ort  # heavy import, load lazily
        from transformers import AutoTokenizer

        directory = onnx_dir or settings.onnx_dir
        self._tokenizer = AutoTokenizer.from_pretrained(directory)
        self._session = ort.InferenceSession(
            str(directory / "model.onnx"), providers=["CPUExecutionProvider"]
        )

    def _encode(self, texts: list[str]) -> np.ndarray:
        enc = self._tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        hidden = self._session.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0]
        mask = enc["attention_mask"][..., None].astype(hidden.dtype)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [f"passage: {t}" for t in texts[start : start + batch_size]]
            out.extend(self._encode(batch).tolist())
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0].tolist()
