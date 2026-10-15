"""The retriever as an MLflow pyfunc model: what the registry stores and the service loads.

It ranks Civil Code articles for a question, exactly like `experiments.MatrixRetriever`:
passage vectors for one or two text variants, scored by cosine and fused with `single`, `max`
or `weighted`. The vectors come from the experiment cache, so registering a model never
re-embeds the corpus.

This file is logged "as code" (`mlflow.pyfunc.log_model(python_model=<this path>)`), so MLflow
runs it on any machine that loads the model, even one without `legal_rag` installed. It must
therefore stay self-contained: numpy, pandas and (lazily) sentence-transformers only.

    model = mlflow.pyfunc.load_model("models:/legal-rag-retriever@production")
    model.predict(pd.DataFrame({"question": ["ما هو سن الرشد؟"]}), params={"k": 3})
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import set_model

DEFAULT_K = 3
QUERY_PREFIX = "query: "  # multilingual-e5 needs it on queries (and "passage: " on documents)
FUSIONS = ("single", "max", "weighted")
COLUMNS = ["question", "rank", "article_number", "citation", "score"]


def slug(variant: str) -> str:
    """Artifact / cache-file safe name of a text variant ('ar+en' -> 'ar_en')."""
    return variant.replace("+", "_")


def _read_json(path: str) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fuse(sims: Sequence[np.ndarray], fusion: str, weights: Sequence[float] = ()) -> np.ndarray:
    """Combine one similarity vector per variant into one score per article."""
    if fusion == "single":
        return sims[0]
    if fusion == "max":
        return np.maximum.reduce(list(sims))
    if fusion == "weighted":
        return sum(w * s for w, s in zip(weights, sims, strict=True))
    raise ValueError(f"unknown fusion {fusion!r}; choose from {FUSIONS}")


class LegalRetriever(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        self.config: dict = {}
        self._articles: list[dict] = []
        self._matrices: list[np.ndarray] = []
        self._encode: Callable[[str], np.ndarray] | None = None

    # -- loading --------------------------------------------------------------------------

    def load_context(self, context) -> None:
        config = _read_json(context.artifacts["config"])
        articles = _read_json(context.artifacts["articles"])
        matrices = [np.load(context.artifacts[f"vectors_{slug(v)}"]) for v in config["variants"]]
        self.setup(config, articles, matrices)

    def setup(
        self,
        config: dict,
        articles: list[dict],
        matrices: list[np.ndarray],
        encode: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        """Validate and install the pieces. `encode` (raw question -> unit vector) is for tests."""
        if config["fusion"] not in FUSIONS:
            raise ValueError(f"unknown fusion {config['fusion']!r}; choose from {FUSIONS}")
        if len(matrices) != len(config["variants"]):
            raise ValueError("one matrix per variant is required")
        for m in matrices:
            if m.shape[0] != len(articles):
                raise ValueError("each matrix needs one row per indexed article")
        if config["fusion"] == "weighted" and len(config["weights"]) != len(matrices):
            raise ValueError("weighted fusion needs one weight per variant")
        self.config = config
        self._articles = articles
        self._matrices = matrices
        self._encode = encode

    # -- scoring --------------------------------------------------------------------------

    def _embed(self, question: str) -> np.ndarray:
        if self._encode is None:  # heavy import and model load happen on the first question
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.config["embedding_model"])
            self._encode = lambda text: model.encode(
                [QUERY_PREFIX + text], normalize_embeddings=True
            )[0]
        return np.asarray(self._encode(question), dtype=np.float32)

    def scores(self, query: np.ndarray) -> np.ndarray:
        sims = [m @ query for m in self._matrices]  # unit vectors: dot product = cosine
        return fuse(sims, self.config["fusion"], self.config["weights"])

    def retrieve(self, question: str, k: int = DEFAULT_K) -> list[dict]:
        if not 1 <= k <= len(self._articles):
            raise ValueError(f"k must be between 1 and {len(self._articles)}")
        scores = self.scores(self._embed(question))
        return [
            {
                "rank": rank,
                "article_number": self._articles[int(i)]["article_number"],
                "citation": self._articles[int(i)]["citation"],
                "score": float(scores[i]),
            }
            for rank, i in enumerate(np.argsort(-scores)[:k], start=1)
        ]

    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        k = int((params or {}).get("k", DEFAULT_K))
        if isinstance(model_input, pd.DataFrame):
            questions = model_input["question"].astype(str).tolist()
        elif isinstance(model_input, str):
            questions = [model_input]
        else:
            questions = [str(q) for q in model_input]
        rows = [{"question": q, **hit} for q in questions for hit in self.retrieve(q, k)]
        return pd.DataFrame(rows, columns=COLUMNS)


set_model(LegalRetriever())
