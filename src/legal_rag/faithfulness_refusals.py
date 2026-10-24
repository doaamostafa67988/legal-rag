"""Split the faithfulness results into answered and refused questions (reads the cache only).

    uv run python -m legal_rag.faithfulness_refusals --lang en --n 15

A refusal ("لا تحتوي المواد المرفقة على ...") makes a negative claim that cannot be checked against
the articles, so the judge scores it oddly (0.0, 0.5, 0.8 for the same behaviour). This report keeps
refusals apart: when the right article was not retrieved, refusing is the correct behaviour, and the
number worth watching is faithfulness over the answers that were actually given. Nothing is sent to
Groq: answers and scores come from the cache written by legal_rag.faithfulness.
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from legal_rag.config import settings
from legal_rag.evaluate import Retriever, load_questions, load_rag
from legal_rag.faithfulness import (
    DEFAULT_CACHE,
    DEFAULT_JUDGE,
    Cache,
    answer_key,
    cited_articles,
    contexts_of,
    score_key,
    strip_citations,
)
from legal_rag.generate import build_prompt

_REFUSAL = re.compile(r"^\s*لا\s+(?:تحتوي|تتضمن|تتناول|توجد|يوجد)\b.{0,40}المواد\s+المرفقة")


def is_refusal(answer: str) -> bool:
    """The answer says the attached articles do not cover the question."""
    return bool(_REFUSAL.match(strip_citations(answer).replace("*", "")))


@dataclass
class RefusalRow:
    id: str
    status: str  # "answered" | "refused-correct" | "refused-wrong" | "missing"
    faithfulness: float | None = None
    cited: tuple[int, ...] = ()
    retrieved: tuple[int, ...] = ()
    gold: tuple[int, ...] = ()
    note: str = ""


def classify(question, hits: list, cache: Cache, *, generator: str, judge: str) -> RefusalRow:
    retrieved = tuple(h.article_number for h in hits)
    gold = tuple(question.gold_articles)
    row = RefusalRow(id=question.id, status="missing", retrieved=retrieved, gold=gold)
    entry = cache.answers.get(answer_key(generator, build_prompt(question.question, hits)))
    if entry is None:
        row.note = "no cached answer: run legal_rag.faithfulness first"
        return row
    answer = entry["answer"]
    body = strip_citations(answer)
    score = cache.scores.get(score_key(judge, question.question, body, contexts_of(hits), True))
    if score is None:
        row.note = "no cached score: run legal_rag.faithfulness first"
        return row
    row.faithfulness = score["faithfulness"]
    row.cited = tuple(cited_articles(answer))
    if not is_refusal(answer):
        row.status = "answered"
    elif set(gold) & set(retrieved):
        row.status = "refused-wrong"  # the article was there and the model still refused
    else:
        row.status = "refused-correct"
    return row


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise_refusals(rows: list[RefusalRow]) -> dict:
    scored = [r for r in rows if r.faithfulness is not None]
    answered = [r.faithfulness for r in scored if r.status == "answered"]
    refused = [r.faithfulness for r in scored if r.status.startswith("refused")]
    n = len(scored)
    return {
        "questions": len(rows),
        "missing": len(rows) - n,
        "answered": len(answered),
        "refused": len(refused),
        "refused_correct": sum(r.status == "refused-correct" for r in rows),
        "refused_wrong": sum(r.status == "refused-wrong" for r in rows),
        "refusal_rate": len(refused) / n if n else None,
        "mean_all": _mean([r.faithfulness for r in scored]),
        "mean_answered": _mean(answered),
        "mean_refused": _mean(refused),
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_row(row: RefusalRow) -> str:
    if row.status == "missing":
        return f"[{row.id}] MISSING {row.note}\n"
    return (
        f"[{row.id}] {row.status:<15} faithfulness={_fmt(row.faithfulness)} "
        f"gold={list(row.gold)} retrieved={list(row.retrieved)} cited={list(row.cited)}\n"
    )


def format_summary(summary: dict) -> str:
    rate = summary["refusal_rate"]
    pct = "n/a" if rate is None else f"{rate:.0%}"
    return (
        f"\nquestions: {summary['questions']}  answered: {summary['answered']}"
        f"  refused: {summary['refused']} ({pct})  missing: {summary['missing']}\n"
        f"refusals, gold article NOT retrieved (correct): {summary['refused_correct']}\n"
        f"refusals, gold article WAS retrieved (wrong):   {summary['refused_wrong']}\n"
        f"faithfulness  answered: {_fmt(summary['mean_answered'])}"
        f"   refused: {_fmt(summary['mean_refused'])}   all: {_fmt(summary['mean_all'])}\n"
    )


def build_rows(questions: list, rag: Retriever, cache: Cache, *, generator: str, judge: str):
    return [
        classify(q, rag.retrieve(q.question, k=3), cache, generator=generator, judge=judge)
        for q in questions
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", default=None, help="only questions of this language, e.g. en")
    ap.add_argument("--ids", nargs="+", default=None, help="only these question ids")
    ap.add_argument("--n", type=int, default=None, help="at most this many questions")
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--questions", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.getLogger("legal_rag").setLevel(logging.ERROR)

    questions = load_questions(args.questions)
    if args.lang:
        questions = [q for q in questions if q.lang == args.lang]
    if args.ids:
        questions = [q for q in questions if q.id in args.ids]
    if args.n:
        questions = questions[: args.n]
    if not questions:
        sys.stderr.write("no question matches the filters\n")
        return 2

    rows = build_rows(
        questions, load_rag(), Cache(args.cache), generator=settings.groq_model, judge=args.judge
    )
    for row in rows:
        sys.stdout.write(format_row(row))
    summary = summarise_refusals(rows)
    sys.stdout.write(format_summary(summary))
    return 0 if not summary["missing"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
