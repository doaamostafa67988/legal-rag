"""Negative control for the faithfulness metric: does the judge notice an ungrounded answer?

    uv run python -m legal_rag.faithfulness_control --lang ar --n 15

Every cached answer is scored again, but against the articles retrieved for a *different* question
(one that shares no article with this answer's retrieved or cited articles). Such an answer is not
grounded in that context, so a metric that measures grounding must score it low. The rule is fixed
before the run: the metric discriminates when the mean control score is below the gate and at most
MAX_HIGH control scores reach it. Nothing is sent to the generator, and the real scores are read
from the cache, not recomputed. Exit code: 0 discriminates, 1 does not, 2 nothing to score.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from legal_rag.config import settings
from legal_rag.evaluate import Retriever, load_questions, load_rag
from legal_rag.faithfulness import (
    DEFAULT_CACHE,
    DEFAULT_JUDGE,
    THRESHOLD,
    Cache,
    Judge,
    RagasJudge,
    answer_key,
    cited_articles,
    contexts_of,
    score_key,
    strip_citations,
)
from legal_rag.generate import build_prompt

MAX_HIGH = 2  # control scores allowed to reach the gate before the metric is called blind


@dataclass
class ControlScore:
    id: str
    foreign_articles: list[int] = field(default_factory=list)
    real: float | None = None
    control: float | None = None
    error: str | None = None
    live_tokens: int = 0


def foreign_pairs(own: list[set[int]], pool: list[set[int]]) -> list[int | None]:
    """For each question, the index of another question whose retrieved articles share nothing with
    `own[i]` (the articles retrieved or cited for it); None when nobody qualifies.

    Candidates are tried in the order i+1, i+2, ... (wrapping), so the choice is deterministic.
    """
    n = len(own)
    partners: list[int | None] = []
    for i in range(n):
        choice = None
        for step in range(1, n):
            j = (i + step) % n
            if not own[i] & pool[j]:
                choice = j
                break
        partners.append(choice)
    return partners


async def run_control(
    questions: list,
    rag: Retriever,
    judge: Judge,
    cache: Cache,
    *,
    generator: str,
    pause: float = 15.0,
    k: int = 3,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_score: Callable[[ControlScore], None] | None = None,
) -> list[ControlScore]:
    hits = [rag.retrieve(q.question, k=k) for q in questions]
    retrieved = [{h.article_number for h in hs} for hs in hits]
    answers: list[str | None] = []
    for q, hs in zip(questions, hits, strict=True):
        entry = cache.answers.get(answer_key(generator, build_prompt(q.question, hs)))
        answers.append(entry["answer"] if entry else None)
    own = [retrieved[i] | set(cited_articles(a or "")) for i, a in enumerate(answers)]
    partners = foreign_pairs(own, retrieved)

    scores: list[ControlScore] = []
    for i, q in enumerate(questions):
        score = ControlScore(id=q.id)
        scores.append(score)
        answer, partner = answers[i], partners[i]
        if answer is None:
            score.error = "no cached answer: run legal_rag.faithfulness first"
        elif partner is None:
            score.error = "no other question shares no article with this one"
        else:
            body = strip_citations(answer)
            real_key = score_key(judge.name, q.question, body, contexts_of(hits[i]), True)
            real = cache.scores.get(real_key)
            score.real = real["faithfulness"] if real else None
            foreign = contexts_of(hits[partner])
            score.foreign_articles = sorted(retrieved[partner])
            key = score_key(judge.name, q.question, body, foreign, True)
            if key in cache.scores:
                score.control = cache.scores[key]["faithfulness"]
            else:
                try:
                    judged = await judge.score(q.question, body, foreign)
                except Exception as exc:  # reported, never cached, does not stop the run
                    score.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                else:
                    score.control = judged.value
                    score.live_tokens = judged.prompt_tokens + judged.completion_tokens
                    cache.scores[key] = {
                        "question_id": f"{q.id}~control",
                        "judge": judge.name,
                        "stripped": True,
                        "scored_answer": body,
                        "faithfulness": judged.value,
                        "prompt_tokens": judged.prompt_tokens,
                        "completion_tokens": judged.completion_tokens,
                    }
                    cache.save()
                await sleep(pause)
        if on_score:
            on_score(score)
    return scores


def summarise_control(scores: list[ControlScore], gate: float = THRESHOLD) -> dict:
    done = [s for s in scores if s.control is not None]
    real = [s.real for s in done if s.real is not None]
    mean_control = sum(s.control for s in done) / len(done) if done else None
    high = [s.id for s in done if s.control >= gate]
    failed = len(scores) - len(done)
    informative = None
    if scores and not failed:
        informative = mean_control < gate and len(high) <= MAX_HIGH
    return {
        "questions": len(scores),
        "scored": len(done),
        "failed": failed,
        "mean_control": mean_control,
        "mean_real": sum(real) / len(real) if real else None,
        "high": high,
        "informative": informative,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_control(score: ControlScore) -> str:
    if score.error:
        return f"[{score.id}] ERROR {score.error}\n"
    return (
        f"[{score.id}] real={_fmt(score.real)} control={_fmt(score.control)} "
        f"(context borrowed from articles {score.foreign_articles})\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", default=None, help="only questions of this language, e.g. ar")
    ap.add_argument("--ids", nargs="+", default=None, help="only these question ids")
    ap.add_argument("--n", type=int, default=None, help="at most this many questions")
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--max-tokens", type=int, default=4096, help="output cap per judge call")
    ap.add_argument("--reasoning-effort", default=None, help="low | medium | high (gpt-oss)")
    ap.add_argument("--pause", type=float, default=15.0, help="seconds after each live score")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
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
    if len(questions) < 2:
        sys.stderr.write("need at least two questions to borrow a context from\n")
        return 2

    judge = RagasJudge(args.judge, args.max_tokens, args.reasoning_effort)
    scores = asyncio.run(
        run_control(
            questions,
            load_rag(),
            judge,
            Cache(args.cache),
            generator=settings.groq_model,
            pause=args.pause,
            on_score=lambda s: sys.stdout.write(format_control(s)),
        )
    )
    summary = summarise_control(scores, args.threshold)
    sys.stdout.write(
        f"\nquestions: {summary['questions']}  scored: {summary['scored']}"
        f"  failed: {summary['failed']}\n"
        f"mean faithfulness, own context (real):        {_fmt(summary['mean_real'])}\n"
        f"mean faithfulness, someone else's (control):  {_fmt(summary['mean_control'])}"
        f"  (gate {args.threshold})\n"
        f"control scores at or above the gate: {summary['high']}\n"
    )
    if summary["informative"] is None:
        sys.stdout.write("verdict: not conclusive (some scores failed or nothing was scored)\n")
        return 2
    sys.stdout.write(
        "verdict: the metric DISCRIMINATES (an ungrounded answer scores low)\n"
        if summary["informative"]
        else "verdict: the metric is BLIND to grounding: do not trust it as a gate\n"
    )
    return 0 if summary["informative"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
