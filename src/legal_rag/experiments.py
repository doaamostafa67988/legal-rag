"""Retrieval experiments tracked in MLflow: how should an article be represented in the index?

    uv run python -m legal_rag.experiments                      # the full sweep, logged to MLflow
    uv run python -m legal_rag.experiments --dry-run            # same numbers, nothing logged
    uv run python -m legal_rag.experiments --models intfloat/multilingual-e5-base

Hold-out check (run once, after the config is chosen on the dev questions):

    uv run python -m legal_rag.experiments --questions data/eval/holdout.json \
        --only "ar" "ar and en vectors, max" --split holdout --experiment legal-rag-holdout

Embedding the corpus is the slow part (about 4 minutes per pass with e5-small on a CPU), so every
(model, text variant) pass is cached in data/cache/. Every ranking experiment on top of the cached
vectors (fusion, weights) is then instant.
"""

import argparse
import contextlib
import hashlib
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from legal_rag.config import ROOT, settings
from legal_rag.data import Article, load_corpus
from legal_rag.evaluate import DEFAULT_QUESTIONS, evaluate, load_questions
from legal_rag.rag import E5Embedder, Embedder, Hit

DEFAULT_CACHE = ROOT / "data" / "cache"
EXPERIMENT = "legal-rag-retrieval"
VARIANTS = ("ar", "en", "ar+en")  # what is embedded for an article
ALPHAS = tuple(round(i / 10, 1) for i in range(11))  # weight of the Arabic vector in a fusion


# ---- what is embedded ---------------------------------------------------------------------


def active_articles(articles: list[Article]) -> list[Article]:
    """The articles that are indexed: same rule as `build_index`."""
    return [a for a in articles if not a.is_repealed and a.text_ar.strip()]


def variant_text(article: Article, variant: str) -> str:
    if variant == "ar":
        return article.text_ar
    if variant == "en":  # a few articles have no English text in the source PDF
        return article.text_en if article.text_en.strip() else article.text_ar
    if variant == "ar+en":
        return f"{article.text_ar}\n{article.text_en}".strip()
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


def _digest(texts: list[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def embedded_corpus(
    embedder: Embedder, model_name: str, articles: list[Article], variant: str, cache_dir: Path
) -> np.ndarray:
    """Passage vectors for one (model, variant), computed once and cached on disk.

    The cache file name carries a digest of the texts, so editing the corpus invalidates it.
    """
    texts = [variant_text(a, variant) for a in active_articles(articles)]
    name = f"{model_name.replace('/', '--')}__{variant.replace('+', '_')}__{_digest(texts)}.npy"
    path = cache_dir / name
    if path.exists():
        return np.load(path)
    vectors = np.asarray(embedder.embed_passages(texts), dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


class MemoEmbedder:
    """Wraps an embedder and remembers query vectors: each question is embedded once per model."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._queries: dict[str, list[float]] = {}

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        if text not in self._queries:
            self._queries[text] = self._embedder.embed_query(text)
        return self._queries[text]


# ---- ranking over cached vectors ----------------------------------------------------------


class MatrixRetriever:
    """In-memory cosine retrieval over cached vectors; same interface as `LegalRAG.retrieve`.

    With one matrix it ranks by that matrix. With several (one per variant) `fusion` decides how the
    scores combine: "max" keeps the best of them (like indexing one chunk per variant),
    "weighted" takes a weighted sum.
    """

    def __init__(
        self,
        embedder: Embedder,
        articles: list[Article],
        matrices: list[np.ndarray],
        fusion: str = "single",
        weights: tuple[float, ...] = (),
    ) -> None:
        self._embedder = embedder
        self._articles = active_articles(articles)
        for m in matrices:
            if m.shape[0] != len(self._articles):
                raise ValueError("each matrix needs one row per indexed article")
        self._matrices = matrices
        self._fusion = fusion
        self._weights = weights

    def _scores(self, query: np.ndarray) -> np.ndarray:
        sims = [m @ query for m in self._matrices]  # vectors are L2-normalised: dot = cosine
        if self._fusion == "single":
            return sims[0]
        if self._fusion == "max":
            return np.maximum.reduce(sims)
        if self._fusion == "weighted":
            return sum(w * s for w, s in zip(self._weights, sims, strict=True))
        raise ValueError(f"unknown fusion {self._fusion!r}")

    def retrieve(self, question: str, k: int = 3) -> list[Hit]:
        query = np.asarray(self._embedder.embed_query(question), dtype=np.float32)
        scores = self._scores(query)
        hits = []
        for i in np.argsort(-scores)[:k]:
            article = self._articles[int(i)]
            hits.append(
                Hit(
                    article_number=article.article_number,
                    citation=article.citation,
                    text_ar=article.text_ar,
                    score=float(scores[i]),
                )
            )
        return hits


# ---- the experiments ----------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    name: str
    family: str
    variants: tuple[str, ...]  # which cached matrices to use
    fusion: str  # "single" | "max" | "weighted"
    weights: tuple[float, ...] = ()
    group: str = ""  # configs of one group become nested runs under a parent run


def sweep_configs() -> list[Config]:
    configs = [Config(v, "single-vector", (v,), "single") for v in VARIANTS]
    configs.append(Config("ar and en vectors, max", "two-vector-max", ("ar", "en"), "max"))
    configs += [
        Config(
            f"alpha={a}",
            "two-vector-weighted",
            ("ar", "en"),
            "weighted",
            (a, round(1 - a, 1)),
            group="weighted sweep",
        )
        for a in ALPHAS
    ]
    return configs


def metric_name(name: str) -> str:
    """MLflow metric names cannot contain '@'."""
    return name.replace("@", "_at_")


def flat_metrics(result: dict) -> dict[str, float]:
    metrics = {metric_name(k): v for k, v in result["overall"].items()}
    for lang, per_lang in result["by_lang"].items():
        metrics.update({f"{lang}_{metric_name(k)}": v for k, v in per_lang.items()})
    return metrics


def record(
    mlflow,
    run_name: str,
    config: Config,
    model: str,
    result: dict,
    context: dict[str, str | int],
    nested: bool = False,
) -> None:
    overall = result["overall"]
    sys.stdout.write(f"{run_name:<44} hit@3={overall['hit@3']:.3f}  mrr={overall['mrr']:.3f}\n")
    if mlflow is None:  # dry run
        return
    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.log_params(
            {
                "embedding_model": model,
                "variants": ",".join(config.variants),
                "fusion": config.fusion,
                "weights": ",".join(str(w) for w in config.weights),
                **context,
            }
        )
        mlflow.set_tag("family", config.family)
        mlflow.log_metrics(flat_metrics(result))
        mlflow.log_dict(result["misses"], "misses.json")


def _run(mlflow, run_name: str):
    if mlflow is None:
        return contextlib.nullcontext()
    return mlflow.start_run(run_name=run_name)


def run_model(
    mlflow,
    model: str,
    embedder: MemoEmbedder,
    articles: list[Article],
    questions: list,
    cache_dir: Path,
    context: dict[str, str | int],
    only: list[str] | None = None,
) -> None:
    """Evaluate the configs for one embedding model; `mlflow=None` prints without logging.

    `only` keeps just the configs with those names (used for the one-shot hold-out run).
    """
    matrices: dict[str, np.ndarray] = {}

    def matrix(variant: str) -> np.ndarray:
        if variant not in matrices:
            matrices[variant] = embedded_corpus(embedder, model, articles, variant, cache_dir)
        return matrices[variant]

    def score(config: Config) -> dict:
        retriever = MatrixRetriever(
            embedder, articles, [matrix(v) for v in config.variants], config.fusion, config.weights
        )
        return evaluate(retriever, questions)

    short = model.split("/")[-1].removeprefix("multilingual-")
    configs = [c for c in sweep_configs() if only is None or c.name in only]
    for config in (c for c in configs if not c.group):
        record(mlflow, f"{short} / {config.name}", config, model, score(config), context)
    for group in dict.fromkeys(c.group for c in configs if c.group):
        best: tuple[tuple[float, float], Config] | None = None
        with _run(mlflow, f"{short} / {group}"):
            for config in (c for c in configs if c.group == group):
                result = score(config)
                record(mlflow, f"{short} / {config.name}", config, model, result, context, True)
                key = (result["overall"]["hit@3"], result["overall"]["mrr"])
                if best is None or key > best[0]:
                    best = (key, config)
            if mlflow is not None and best is not None:
                mlflow.log_params({"embedding_model": model, "best_alpha": best[1].weights[0]})
                mlflow.log_metrics({"best_hit_at_3": best[0][0], "best_mrr": best[0][1]})


def file_sha256(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m legal_rag.experiments")
    ap.add_argument(
        "--models",
        nargs="+",
        default=[settings.embedding_model],
        help="e5-family models only: they need the query:/passage: prefixes",
    )
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--experiment", default=EXPERIMENT)
    ap.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="run only the configs with these names (use it for the hold-out run)",
    )
    ap.add_argument("--split", default="dev", help="label logged as a param: dev or holdout")
    ap.add_argument("--dry-run", action="store_true", help="evaluate and print, log nothing")
    args = ap.parse_args(argv)
    logging.getLogger("legal_rag").setLevel(logging.ERROR)

    known = [c.name for c in sweep_configs()]
    unknown = [name for name in args.only or [] if name not in known]
    if unknown:
        ap.error(f"unknown config(s) {unknown}; choose from {known}")

    mlflow = None
    if not args.dry_run:
        import mlflow as mlflow_module

        mlflow_module.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow_module.set_experiment(args.experiment)
        mlflow = mlflow_module

    articles = load_corpus()
    questions = load_questions(args.questions)
    context: dict[str, str | int] = {
        "corpus_sha256": file_sha256(settings.corpus_json),
        "eval_set_sha256": file_sha256(args.questions),
        "n_docs": len(active_articles(articles)),
        "n_questions": len(questions),
        "split": args.split,
        "git_commit": git_commit(),
        # one chunk per article, so there is no window size or overlap to tune
        "chunk_strategy": "article",
        "chunk_size": "article",
        "overlap": 0,
    }
    for model in args.models:
        embedder = MemoEmbedder(E5Embedder(model))
        run_model(
            mlflow, model, embedder, articles, questions, args.cache_dir, context, only=args.only
        )


if __name__ == "__main__":
    main()
