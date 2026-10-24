"""Register a finalist run as a version of the model `legal-rag-retriever`.

    uv run python -m legal_rag.register --run-name "e5-base / ar and en vectors, max"   # version 1
    uv run python -m legal_rag.register --run-name "e5-small / ar"                      # version 2
    uv run python -m legal_rag.registry show

What is registered is the *retriever*: the config the finalist run logged (embedding model, text
variants, fusion, weights), the passage vectors from the experiment cache and the article list
those vectors are rows of, wrapped by `retriever_model.LegalRetriever`. The model is logged INTO
the finalist run, so the version links back to the metrics, params, git commit and data version
that justified it. A version gets no alias here: that is the "None" stage. Promotion is separate
(`legal_rag.registry`).

Registering again a run that already has a version prints that version and does nothing.
"""

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from legal_rag.registry import NAME, version_for_run
from legal_rag.retriever_model import slug

FINALISTS_EXPERIMENT = "legal-rag-finalists"
MODEL_FILE = Path(__file__).with_name("retriever_model.py")
PIP_REQUIREMENTS = ["numpy", "pandas", "sentence-transformers"]


@dataclass(frozen=True)
class RetrieverSpec:
    embedding_model: str
    variants: tuple[str, ...]
    fusion: str
    weights: tuple[float, ...]
    corpus_sha256: str
    source_run_id: str
    source_run_name: str


def spec_from_params(params: dict[str, str], run_id: str, run_name: str) -> RetrieverSpec:
    """Read the retriever config back from the params a finalist run logged."""
    return RetrieverSpec(
        embedding_model=params["embedding_model"],
        variants=tuple(v for v in params["variants"].split(",") if v),
        fusion=params["fusion"],
        weights=tuple(float(w) for w in params.get("weights", "").split(",") if w),
        corpus_sha256=params["corpus_sha256"],
        source_run_id=run_id,
        source_run_name=run_name,
    )


class _NoEmbedder:
    """Registering never embeds: a missing cache file means the experiment was not run here."""

    def embed_passages(self, texts):
        raise FileNotFoundError(
            "vectors are not in data/cache/: run the experiment on this machine first"
        )

    def embed_query(self, text):
        raise NotImplementedError


def build_bundle(spec: RetrieverSpec, out_dir: Path, cache_dir: Path) -> dict[str, str]:
    """Write config.json, articles.json and one vectors_<variant>.npy; return the artifact map."""
    import numpy as np

    from legal_rag.config import settings
    from legal_rag.data import load_corpus
    from legal_rag.experiments import active_articles, embedded_corpus, file_sha256

    current = file_sha256(settings.corpus_json)
    if current != spec.corpus_sha256:
        raise ValueError(
            f"the corpus changed since run {spec.source_run_id[:8]} "
            f"({spec.corpus_sha256[:12]} -> {current[:12]}): its vectors no longer match"
        )
    articles = load_corpus()
    active = active_articles(articles)
    artifacts: dict[str, str] = {}
    for variant in spec.variants:
        vectors = embedded_corpus(_NoEmbedder(), spec.embedding_model, articles, variant, cache_dir)
        path = out_dir / f"vectors_{slug(variant)}.npy"
        np.save(path, vectors)
        artifacts[f"vectors_{slug(variant)}"] = str(path)

    config = {
        "embedding_model": spec.embedding_model,
        "variants": list(spec.variants),
        "fusion": spec.fusion,
        "weights": list(spec.weights),
        "corpus_sha256": spec.corpus_sha256,
        "chunk_strategy": "article",
        "n_articles": len(active),
        "source_run_id": spec.source_run_id,
        "source_run_name": spec.source_run_name,
    }
    rows = [{"article_number": a.article_number, "citation": a.citation} for a in active]
    for key, payload in (("config", config), ("articles", rows)):
        path = out_dir / f"{key}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts[key] = str(path)
    return artifacts


def find_run(client, experiment: str, run_name: str):
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise LookupError(f"no experiment named {experiment!r}")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise LookupError(f"no run named {run_name!r} in {experiment!r}")
    if len(runs) > 1:
        sys.stderr.write(f"{len(runs)} runs named {run_name!r}: using the most recent\n")
    return runs[0]


def register(run_name: str, experiment: str = FINALISTS_EXPERIMENT, cache_dir: Path | None = None):
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import ModelSignature
    from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

    from legal_rag.config import settings
    from legal_rag.experiments import DEFAULT_CACHE

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    run = find_run(client, experiment, run_name)
    existing = version_for_run(client, run.info.run_id)
    if existing is not None:
        return existing

    spec = spec_from_params(run.data.params, run.info.run_id, run_name)
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "question")]),
        outputs=Schema(
            [
                ColSpec("string", "question"),
                ColSpec("long", "rank"),
                ColSpec("long", "article_number"),
                ColSpec("string", "citation"),
                ColSpec("double", "score"),
            ]
        ),
        params=ParamSchema([ParamSpec("k", "long", 3)]),
    )
    before = {int(v.version) for v in client.search_model_versions(f"name='{NAME}'")}
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = build_bundle(spec, Path(tmp), cache_dir or DEFAULT_CACHE)
        with mlflow.start_run(run_id=run.info.run_id):  # log into the finalist run itself
            mlflow.pyfunc.log_model(
                name="retriever",
                python_model=str(MODEL_FILE),
                artifacts=artifacts,
                signature=signature,
                pip_requirements=PIP_REQUIREMENTS,
                registered_model_name=NAME,
            )
    after = client.search_model_versions(f"name='{NAME}'")
    created = [v for v in after if int(v.version) not in before]
    if len(created) != 1:
        raise RuntimeError(f"expected one new version of {NAME!r}, found {len(created)}")
    version = created[0]
    for key, value in {
        "source_run_id": run.info.run_id,
        "source_run_name": run_name,
        "embedding_model": spec.embedding_model,
        "fusion": spec.fusion,
    }.items():
        client.set_model_version_tag(NAME, version.version, key, value)
    return client.get_model_version(NAME, version.version)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m legal_rag.register")
    ap.add_argument("--run-name", required=True, help='e.g. "e5-base / ar and en vectors, max"')
    ap.add_argument("--experiment", default=FINALISTS_EXPERIMENT)
    ap.add_argument("--cache-dir", type=Path)
    args = ap.parse_args(argv)
    version = register(args.run_name, args.experiment, args.cache_dir)
    sys.stdout.write(f"{NAME} v{version.version} <- run {args.run_name!r}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
