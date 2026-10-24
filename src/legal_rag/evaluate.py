"""Retrieval evaluation: hit@k and MRR on the question set, overall and per language.

    uv run python -m legal_rag.evaluate [--questions data/eval/questions.json] [--k 5]

The question set stores the gold article(s) of every question. A question is a hit at k when any
gold article is among the first k distinct articles retrieved.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from legal_rag.config import ROOT, settings
from legal_rag.rag import Hit, LegalRAG

DEFAULT_QUESTIONS = ROOT / "data" / "eval" / "questions.json"
KS = (1, 3, 5)
HEADLINE_K = 3  # the number quoted in reports; misses are listed against it


class Retriever(Protocol):
    def retrieve(self, question: str, k: int = 3) -> list[Hit]: ...


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    lang: str
    question: str
    gold_articles: tuple[int, ...]


def load_questions(path: Path | None = None) -> list[EvalQuestion]:
    with open(path or DEFAULT_QUESTIONS, encoding="utf-8") as f:
        records = json.load(f)
    return [
        EvalQuestion(
            id=r["id"],
            lang=r["lang"],
            question=r["question"],
            gold_articles=tuple(r["gold_articles"]),
        )
        for r in records
    ]


def _first_gold_rank(question: EvalQuestion, retrieved: list[int]) -> int | None:
    """1-based rank of the first gold article in the retrieved list, or None."""
    for rank, number in enumerate(retrieved, start=1):
        if number in question.gold_articles:
            return rank
    return None


def _summarise(ranks: list[int | None], ks: tuple[int, ...]) -> dict[str, float]:
    if not ranks:
        return {}
    n = len(ranks)
    summary = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    summary["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return summary


def evaluate(rag: Retriever, questions: list[EvalQuestion], ks: tuple[int, ...] = KS) -> dict:
    """Run every question through `rag.retrieve` and score the ranking against the gold articles."""
    if HEADLINE_K not in ks:
        raise ValueError(f"ks must include {HEADLINE_K}, the headline cut-off")
    k_max = max(ks)
    ranks: dict[str, int | None] = {}
    misses = []
    for q in questions:
        # several chunks of one article must count once, so de-duplicate keeping the order
        retrieved = list(dict.fromkeys(h.article_number for h in rag.retrieve(q.question, k=k_max)))
        rank = _first_gold_rank(q, retrieved)
        ranks[q.id] = rank
        if rank is None or rank > HEADLINE_K:
            misses.append(
                {
                    "id": q.id,
                    "question": q.question,
                    "gold": list(q.gold_articles),
                    "retrieved": retrieved[:HEADLINE_K],
                }
            )
    return {
        "n": len(questions),
        "overall": _summarise(list(ranks.values()), ks),
        "by_lang": {
            lang: _summarise([ranks[q.id] for q in questions if q.lang == lang], ks)
            for lang in sorted({q.lang for q in questions})
        },
        "misses": misses,
    }


def format_report(result: dict) -> str:
    def row(metrics: dict[str, float]) -> str:
        return "  ".join(f"{name}={value:.3f}" for name, value in metrics.items())

    lines = [f"questions: {result['n']}", f"{'overall':<8} {row(result['overall'])}"]
    lines += [f"{lang:<8} {row(metrics)}" for lang, metrics in result["by_lang"].items()]
    if result["misses"]:
        lines.append(f"misses (gold not in the first {HEADLINE_K} articles):")
        lines += [
            f"  {m['id']}: gold {m['gold']}, retrieved {m['retrieved']}" for m in result["misses"]
        ]
    return "\n".join(lines) + "\n"


def load_rag(backend: str | None = None) -> LegalRAG:
    """Build the retriever the way the API does, so the numbers describe what is served."""
    backend = backend or settings.embedder_backend
    if backend == "onnx":
        from legal_rag.onnx_embed import OnnxEmbedder

        return LegalRAG(embedder=OnnxEmbedder())
    return LegalRAG()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m legal_rag.evaluate")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--backend", choices=["onnx", "torch"], default=None)
    ap.add_argument("--json", action="store_true", help="print the raw result as JSON")
    args = ap.parse_args(argv)
    logging.getLogger("legal_rag").setLevel(logging.ERROR)  # keep the report free of request logs
    result = evaluate(load_rag(args.backend), load_questions(args.questions))
    if args.json:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(format_report(result))


if __name__ == "__main__":
    main()
