"""Serve the retriever the model registry says is in production (Module 2, Step 3).

`LegalRAG` searches a Chroma index baked into the image: one Arabic vector per article, i.e. the
"e5-small / ar" config. `RegistryRAG` has the same surface (`retrieve`, `documents_indexed`,
`embedder`, `get_article`) but ranks with whatever config is registered, by default
`models:/legal-rag-retriever@production`: the vectors, text variants and fusion of the run that
justified it. Moving the alias in the MLflow UI and restarting the container changes what is
served; no code changes and no image rebuild.

Queries are still embedded here, by the ONNX encoder the service already ships, so a registered
model is only servable if it was built with that same embedding model (checked at startup).
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from legal_rag.config import settings
from legal_rag.data import Article, load_corpus
from legal_rag.rag import Embedder, Hit, _metadata
from legal_rag.timing import timed

log = logging.getLogger(__name__)

FUSIONS = ("single", "max", "weighted")
_URI = re.compile(r"models:/(?P<name>[^/@]+)(?:@(?P<alias>[^/@]+)|/(?P<version>\d+))")


class ModelMismatchError(RuntimeError):
    """The registered model cannot be served by this build: say why, and refuse to start."""


@dataclass(frozen=True)
class Bundle:
    """What `legal_rag.register` stored: config.json, articles.json and one matrix per variant."""

    config: dict
    articles: list[dict]
    matrices: list[np.ndarray]


def slug(variant: str) -> str:
    return variant.replace("+", "_")  # same rule as experiments' cache files and retriever_model


def fuse(sims: list[np.ndarray], fusion: str, weights: list[float]) -> np.ndarray:
    """Same three fusions as `experiments.MatrixRetriever` and `retriever_model.fuse`."""
    if fusion == "single":
        return sims[0]
    if fusion == "max":
        return np.maximum.reduce(sims)
    if fusion == "weighted":
        return sum(w * s for w, s in zip(weights, sims, strict=True))
    raise ValueError(f"unknown fusion {fusion!r}; choose from {FUSIONS}")


def log_fields(info: dict) -> dict:
    """Prefix the keys: `extra=` may not reuse a LogRecord attribute, and `name` is one."""
    return {f"model_{key}": value for key, value in info.items()}


def _find(directory: Path, name: str) -> Path:
    # a downloaded pyfunc model keeps its artifacts under artifacts/; tolerate other layouts
    matches = sorted(directory.rglob(name), key=lambda p: p.parent.name != "artifacts")
    if not matches:
        raise FileNotFoundError(f"{name} not found under {directory}")
    return matches[0]


def load_bundle(directory: Path) -> Bundle:
    config = json.loads(_find(directory, "config.json").read_text(encoding="utf-8"))
    articles = json.loads(_find(directory, "articles.json").read_text(encoding="utf-8"))
    matrices = [np.load(_find(directory, f"vectors_{slug(v)}.npy")) for v in config["variants"]]
    return Bundle(config, articles, matrices)


def parse_uri(uri: str) -> tuple[str, str | None, str | None]:
    """`models:/name@alias` or `models:/name/3` -> (name, alias, version)."""
    m = _URI.fullmatch(uri)
    if m is None:
        raise ValueError(f"expected models:/<name>@<alias> or models:/<name>/<n>, got {uri!r}")
    return m["name"], m["alias"], m["version"]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_compatible(bundle: Bundle, corpus_json: Path | None = None) -> None:
    """Refuse a model this build cannot serve correctly. Raises ModelMismatchError."""
    config = bundle.config
    if config["embedding_model"] != settings.embedding_model:
        raise ModelMismatchError(
            f"the registered model was built with {config['embedding_model']!r} but this service "
            f"encodes queries with {settings.embedding_model!r} (the ONNX encoder in "
            f"{settings.onnx_dir}). Export the registered model's encoder with legal_rag.export, "
            "set LEGAL_RAG_EMBEDDING_MODEL and LEGAL_RAG_ONNX_DIR, or move the alias back."
        )
    on_disk = _sha256(corpus_json or settings.corpus_json)
    if on_disk != config["corpus_sha256"]:
        raise ModelMismatchError(
            "the corpus in this image is not the one the registered vectors were built from "
            f"(registered {config['corpus_sha256'][:12]}, here {(on_disk or 'missing')[:12]}); "
            "the vectors would not match the article texts"
        )


class RegistryRAG:
    """Query side over a registered bundle; a drop-in for `LegalRAG` in the API."""

    def __init__(
        self,
        bundle: Bundle,
        embedder: Embedder,
        articles: list[Article] | None = None,
        info: dict | None = None,
        low_score_threshold: float | None = None,
    ) -> None:
        config = bundle.config
        if config["fusion"] not in FUSIONS:
            raise ValueError(f"unknown fusion {config['fusion']!r}; choose from {FUSIONS}")
        for m in bundle.matrices:
            if m.shape[0] != len(bundle.articles):
                raise ValueError("each matrix needs one row per registered article")
        if config["fusion"] == "weighted" and len(config["weights"]) != len(bundle.matrices):
            raise ValueError("weighted fusion needs one weight per variant")
        corpus = articles if articles is not None else load_corpus()
        self._articles = {a.article_number: a for a in corpus}
        numbers = [r["article_number"] for r in bundle.articles]
        missing = [n for n in numbers if n not in self._articles]
        if missing:
            raise ModelMismatchError(f"registered articles missing from the corpus: {missing[:5]}")
        self._rows = [self._articles[n] for n in numbers]
        self._matrices = bundle.matrices
        self._fusion = config["fusion"]
        self._weights = config["weights"]
        self.embedder = embedder
        self.info = {"source": "registry", **(info or {})}
        self._low_score_threshold = (
            settings.low_score_threshold if low_score_threshold is None else low_score_threshold
        )

    @property
    def documents_indexed(self) -> int:
        return len(self._rows)  # used by /health

    @timed
    def retrieve(self, question: str, k: int = 3) -> list[Hit]:
        if not question.strip():
            log.warning("rejected empty question")
            raise ValueError("question must not be empty")
        start = time.perf_counter()
        query = np.asarray(self.embedder.embed_query(question), dtype=np.float32)
        scores = fuse([m @ query for m in self._matrices], self._fusion, self._weights)
        order = np.argsort(-scores)[: min(k, len(self._rows))]
        hits = [
            Hit(
                article_number=self._rows[i].article_number,
                citation=self._rows[i].citation,
                text_ar=self._rows[i].text_ar,
                score=float(scores[i]),
                metadata=_metadata(self._rows[i]),
            )
            for i in map(int, order)
        ]
        log.info(
            "question served",
            extra={
                "k": k,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "top_article": hits[0].article_number if hits else None,
                "top_score": round(hits[0].score, 3) if hits else None,
                "question_chars": len(question),
                "model_version": self.info.get("version"),
            },
        )
        if hits and hits[0].score < self._low_score_threshold:
            log.warning(
                "low retrieval confidence",
                extra={
                    "top_score": round(hits[0].score, 3),
                    "threshold": self._low_score_threshold,
                },
            )
        return hits

    def get_article(self, number: int) -> Article | None:
        return self._articles.get(number)


def load_from_registry(uri: str, embedder: Embedder, cache_dir: Path | None = None) -> RegistryRAG:
    """Resolve the alias, download that version once, check it fits this build, load it."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    name, alias, version = parse_uri(uri)
    client = MlflowClient()
    mv = (
        client.get_model_version_by_alias(name, alias)
        if alias
        else client.get_model_version(name, version)
    )
    # versions are immutable, so a downloaded version is reused; a moved alias is a new directory
    target = (cache_dir or settings.registry_cache_dir) / name / str(mv.version)
    if not any(target.rglob("config.json")):
        target.mkdir(parents=True, exist_ok=True)
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{name}/{mv.version}", dst_path=str(target)
        )
    bundle = load_bundle(target)
    check_compatible(bundle)
    info = {
        "name": name,
        "version": str(mv.version),
        "alias": alias,
        "run_id": getattr(mv, "run_id", None) or (mv.tags or {}).get("source_run_id"),
        "embedding_model": bundle.config["embedding_model"],
        "variants": bundle.config["variants"],
        "fusion": bundle.config["fusion"],
    }
    log.info("retriever loaded from the registry", extra=log_fields(info))
    return RegistryRAG(bundle, embedder, info=info)
