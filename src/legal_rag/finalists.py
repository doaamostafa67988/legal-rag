"""Compare the finalist configs on retrieval AND answer faithfulness: one MLflow run each.

    uv run python -m legal_rag.finalists               # plan: what would be generated and judged
    uv run python -m legal_rag.finalists --run         # do it (spends Groq tokens, cached)
    uv run python -m legal_rag.finalists --run --pick 1 2     # only these finalists

The default is the plan: it retrieves, checks the faithfulness cache and prints how many answers
and judge scores are still missing, without calling Groq. Retrieval metrics are computed on every
dev question; faithfulness only on the first --n Arabic ones (the same questions as
legal_rag.faithfulness, so earlier answers and scores are reused), which is why each run also logs
ar_hit_at_3 and ar_mrr. The hold-out is never used here. A finalist whose scoring did not finish
is not logged: run again, the scores already paid for come from the cache.
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from legal_rag import faithfulness as faith
from legal_rag.config import settings
from legal_rag.data import load_corpus
from legal_rag.evaluate import DEFAULT_QUESTIONS, evaluate, load_questions
from legal_rag.experiments import (
    DEFAULT_CACHE,
    Config,
    MatrixRetriever,
    MemoEmbedder,
    RunContext,
    active_articles,
    embedded_corpus,
    file_sha256,
    flat_metrics,
    git_author,
    git_commit,
    index_cost,
    requirements_text,
    sweep_configs,
)
from legal_rag.generate import GroqChat, build_prompt
from legal_rag.rag import E5Embedder

EXPERIMENT = "legal-rag-finalists"
SMALL = "intfloat/multilingual-e5-small"
BASE = "intfloat/multilingual-e5-base"
# chosen on the dev questions only: the baseline and the best of each model, plus a middle fusion
FINALISTS = (
    (SMALL, "ar"),
    (SMALL, "ar and en vectors, max"),
    (BASE, "ar"),
    (BASE, "ar+en"),
    (BASE, "alpha=0.5"),
    (BASE, "ar and en vectors, max"),
)
DEFAULT_SCORE_TOKENS = 3400  # used for the estimate when the cache has no score to average


@dataclass(frozen=True)
class Finalist:
    model: str
    config: Config

    @property
    def name(self) -> str:
        short = self.model.split("/")[-1].removeprefix("multilingual-")
        return f"{short} / {self.config.name}"


def finalist_list() -> list[Finalist]:
    configs = {c.name: c for c in sweep_configs()}
    unknown = [name for _, name in FINALISTS if name not in configs]
    if unknown:
        raise ValueError(f"finalists name configs that do not exist: {unknown}")
    return [Finalist(model, configs[name]) for model, name in FINALISTS]


def pending_work(
    questions: list, hits: list[list], cache: faith.Cache, judge_name: str, generator: str
) -> tuple[set[str], set[str]]:
    """Which answers and judge scores this finalist still needs (as cache keys)."""
    answers_new: set[str] = set()
    scores_new: set[str] = set()
    for q, hs in zip(questions, hits, strict=True):
        akey = faith.answer_key(generator, build_prompt(q.question, hs))
        entry = cache.answers.get(akey)
        if entry is None:  # no answer yet, so its score cannot exist either
            answers_new.add(akey)
            scores_new.add(f"after:{akey}")
            continue
        body = faith.strip_citations(entry["answer"])
        skey = faith.score_key(judge_name, q.question, body, faith.contexts_of(hs), True)
        if skey not in cache.scores:
            scores_new.add(skey)
    return answers_new, scores_new


def average_score_tokens(cache: faith.Cache) -> int:
    costs = [v["prompt_tokens"] + v["completion_tokens"] for v in cache.scores.values()]
    return round(sum(costs) / len(costs)) if costs else DEFAULT_SCORE_TOKENS


def format_plan(rows: list[tuple[str, int, int]], unique: tuple[int, int], per_score: int) -> str:
    lines = [f"{'finalist':<34}{'answers to generate':>21}{'scores to judge':>17}"]
    lines += [f"{name:<34}{a:>21}{s:>17}" for name, a, s in rows]
    answers, scores = unique
    lines.append(f"{'unique across all finalists':<34}{answers:>21}{scores:>17}")
    lines.append(
        f"estimated judge tokens: about {scores * per_score:,} "
        f"({scores} scores x {per_score:,} average from the cache)"
    )
    lines.append("nothing was sent to Groq")
    return "\n".join(lines) + "\n"


def build_context(questions_path: Path, articles: list, questions: list) -> RunContext:
    corpus_hash = file_sha256(settings.corpus_json)
    return RunContext(
        params={
            "corpus_sha256": corpus_hash,
            "eval_set_sha256": file_sha256(questions_path),
            "n_docs": len(active_articles(articles)),
            "n_questions": len(questions),
            "split": "dev",
            "chunk_strategy": "article",
            "chunk_size": "article",
            "overlap": 0,
        },
        tags={
            "git_commit": git_commit(),
            "data_version": corpus_hash[:12],
            "author": git_author(),
            "framework": "sentence-transformers + numpy",
        },
        texts={"requirements.txt": requirements_text()},
    )


def record_finalist(mlflow, finalist, result, cost, summary, scores, ctx, args) -> None:
    config = finalist.config
    metrics = {**flat_metrics(result), **cost}
    metrics.update(
        faithfulness=summary["mean_faithfulness"],
        faithfulness_below_gate=float(len(summary["below_threshold"])),
        citation_valid_rate=summary["citation_valid_rate"],
        judge_tokens_cold=float(summary["cold_tokens"]),
    )
    with mlflow.start_run(run_name=finalist.name):
        mlflow.log_params(
            {
                "embedding_model": finalist.model,
                "variants": ",".join(config.variants),
                "fusion": config.fusion,
                "weights": ",".join(str(w) for w in config.weights),
                "faithfulness_judge": args.judge,
                "faithfulness_lang": "ar",
                "faithfulness_n": summary["n"],
                "faithfulness_gate": args.threshold,
                **ctx.params,
            }
        )
        mlflow.set_tags({"family": config.family, "stage": "finalist", **ctx.tags})
        mlflow.log_metrics(metrics)
        mlflow.log_dict(result["misses"], "misses.json")
        mlflow.log_dict(
            [
                {
                    "id": s.id,
                    "faithfulness": s.faithfulness,
                    "cited": s.cited,
                    "retrieved": s.retrieved,
                }
                for s in scores
            ],
            "faithfulness.json",
        )
        for name, text in ctx.texts.items():
            mlflow.log_text(text, name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m legal_rag.finalists", description=__doc__)
    ap.add_argument("--run", action="store_true", help="generate, judge and log (default: plan)")
    ap.add_argument("--pick", type=int, nargs="+", help="1-based finalists to use (see the plan)")
    ap.add_argument("--n", type=int, default=15, help="Arabic questions to judge")
    ap.add_argument("--force", action="store_true", help="log again even if a run exists")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="embedding cache")
    ap.add_argument("--faith-cache", type=Path, default=faith.DEFAULT_CACHE)
    ap.add_argument("--experiment", default=EXPERIMENT)
    ap.add_argument("--judge", default=faith.DEFAULT_JUDGE)
    ap.add_argument("--max-tokens", type=int, default=4096, help="output cap per judge call")
    ap.add_argument("--reasoning-effort", default=None, help="low | medium | high (gpt-oss)")
    ap.add_argument("--pause", type=float, default=15.0, help="seconds after each live score")
    ap.add_argument("--threshold", type=float, default=faith.THRESHOLD)
    args = ap.parse_args(argv)
    if "holdout" in Path(args.questions).name:
        ap.error("finalists are compared on the dev questions; the hold-out is not used here")
    logging.getLogger("legal_rag").setLevel(logging.ERROR)

    finalists = finalist_list()
    if args.pick:
        bad = [n for n in args.pick if not 1 <= n <= len(finalists)]
        if bad:
            ap.error(f"--pick takes numbers from 1 to {len(finalists)}, got {bad}")
        finalists = [finalists[n - 1] for n in args.pick]

    articles = load_corpus()
    questions = load_questions(args.questions)
    arabic = [q for q in questions if q.lang == "ar"][: args.n]
    if not arabic:
        sys.stderr.write("no Arabic question to judge\n")
        return 2

    embedders: dict[str, MemoEmbedder] = {}
    retrievers = {}
    for f in finalists:
        embedder = embedders.setdefault(f.model, MemoEmbedder(E5Embedder(f.model)))
        used = [
            embedded_corpus(embedder, f.model, articles, v, args.cache_dir)
            for v in f.config.variants
        ]
        retrievers[f.name] = (
            MatrixRetriever(embedder, articles, used, f.config.fusion, f.config.weights),
            index_cost(used),
        )

    cache = faith.Cache(args.faith_cache)
    judge = faith.RagasJudge(args.judge, args.max_tokens, args.reasoning_effort)
    generator = settings.groq_model
    if not args.run:
        rows, answers_all, scores_all = [], set(), set()
        for f in finalists:
            hits = [retrievers[f.name][0].retrieve(q.question, k=3) for q in arabic]
            a, s = pending_work(arabic, hits, cache, judge.name, generator)
            rows.append((f.name, len(a), len(s)))
            answers_all |= a
            scores_all |= s
        unique = (len(answers_all), len(scores_all))
        sys.stdout.write(format_plan(rows, unique, average_score_tokens(cache)))
        return 0

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)
    ctx = build_context(args.questions, articles, questions)
    chat = GroqChat()
    incomplete = 0
    for f in finalists:
        existing = mlflow.search_runs(
            experiment_names=[args.experiment], filter_string=f"tags.mlflow.runName = '{f.name}'"
        )
        if len(existing) and not args.force:
            sys.stdout.write(f"{f.name}: already logged, skipped (--force to log again)\n")
            continue
        retriever, cost = retrievers[f.name]
        judge = faith.RagasJudge(args.judge, args.max_tokens, args.reasoning_effort)
        scores = asyncio.run(
            faith.score_all(
                arabic, retriever, chat, judge, cache,
                generator=generator, pause=args.pause,
            )
        )  # fmt: skip
        summary = faith.summarise(scores, args.threshold)
        if summary["failed"]:
            incomplete += 1
            sys.stdout.write(
                f"{f.name}: {summary['failed']} of {summary['n']} scores failed, not logged; "
                "run again (what was paid for is cached)\n"
            )
            continue
        record_finalist(mlflow, f, evaluate(retriever, questions), cost, summary, scores, ctx, args)
        mean = summary["mean_faithfulness"]
        sys.stdout.write(f"{f.name:<34} faithfulness={mean:.3f}  logged\n")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
