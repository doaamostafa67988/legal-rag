"""Sensitivity check for the faithfulness metric: does the judge notice a corrupted answer?

    uv run python -m legal_rag.faithfulness_perturb --lang ar --n 15

The negative control shows the judge rejects an answer paired with unrelated articles. This asks
the finer question: if a good cached answer is changed in one place (a figure, a time limit or the
ruling itself), does its score drop? Each answer gets one deterministic corruption, the corrupted
text is scored against the *same* articles, and the real score is read from the cache.

The rule is fixed before the run: the metric is sensitive when at least MIN_DETECTED of the
corrupted answers lose MIN_DROP or more. Refusals, answers already below the gate and answers no
rule can corrupt are skipped and reported. Exit code: 0 sensitive, 1 not sensitive, 2 not
conclusive.
"""

import argparse
import asyncio
import logging
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from legal_rag.config import settings
from legal_rag.evaluate import Retriever, load_questions, load_rag
from legal_rag.faithfulness import (
    _CITED,
    _DIGIT_MAP,
    DEFAULT_CACHE,
    DEFAULT_JUDGE,
    THRESHOLD,
    Cache,
    Judge,
    RagasJudge,
    answer_key,
    contexts_of,
    score_key,
    strip_citations,
)
from legal_rag.generate import build_prompt

MIN_DROP = 0.10  # a corrupted answer must score at least this much lower to count as noticed
MIN_DETECTED = (
    0.70  # share of corrupted answers that must be noticed for the metric to be sensitive
)

_FIGURE = re.compile(r"[0-9٠-٩]+")
# "1." or "2)" or "(3)" numbering, also in the middle of a line: strip_citations joins lines
_LIST_MARKER = re.compile(r"(?<![\w%.])[0-9]{1,2}[.)](?=\s)|\(\s*[0-9]+\s*\)")
_CAN = re.compile(r"(?<!لا )يجوز")
_REFUSAL = "لا تحتوي"
# a time limit or a rate written in words, with a wrong value that is still plausible
_PHRASES = (
    ("خمس عشرة سنة", "عشر سنوات"),
    ("ثلاث سنوات", "خمس سنوات"),
    ("إحدى وعشرون سنة", "ثماني عشرة سنة"),
    ("إحدى وعشرين سنة", "ثماني عشرة سنة"),
    ("أربعة في المائة", "سبعة في المائة"),
    ("خمسة في المائة", "عشرة في المائة"),
)


@dataclass(frozen=True)
class Corruption:
    kind: str
    before: str
    after: str
    text: str


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in spans)


def _swap_figure(text: str) -> Corruption | None:
    """Change a figure that is a fact (not an article number, a paragraph or a list marker)."""
    ascii_text = text.translate(_DIGIT_MAP)  # same length, so spans line up with `text`
    skip = [m.span() for m in _CITED.finditer(ascii_text)]
    skip += [m.span() for m in _LIST_MARKER.finditer(ascii_text)]
    for m in _FIGURE.finditer(text):
        if _overlaps(m.span(), skip) or "فقر" in text[max(0, m.start() - 8) : m.start()]:
            continue
        old = m.group().translate(_DIGIT_MAP)
        new = str(int(old) * 2 + 1)
        return Corruption("figure", old, new, text[: m.start()] + new + text[m.end() :])
    return None


def _swap_phrase(text: str) -> Corruption | None:
    for old, new in _PHRASES:
        if old in text:
            return Corruption("time limit or rate", old, new, text.replace(old, new, 1))
    return None


def _flip_ruling(text: str) -> Corruption | None:
    if "لا يجوز" in text:
        return Corruption("ruling", "لا يجوز", "يجوز", text.replace("لا يجوز", "يجوز", 1))
    m = _CAN.search(text)
    if m:
        flipped = text[: m.start()] + "لا يجوز" + text[m.end() :]
        return Corruption("ruling", "يجوز", "لا يجوز", flipped)
    return None


def corrupt(text: str) -> Corruption | None:
    """One deterministic change: a figure, else a time limit or rate, else the ruling.

    A yes/no particle ("نعم" / "لا") is never flipped: faithfulness checks the claims against the
    articles, not whether the answer fits the question, so that change would not be an error to it.
    """
    for rule in (_swap_figure, _swap_phrase, _flip_ruling):
        found = rule(text)
        if found:
            return found
    return None


@dataclass
class PerturbScore:
    id: str
    kind: str = ""
    change: str = ""
    real: float | None = None
    corrupted: float | None = None
    skipped: str | None = None
    error: str | None = None
    live_tokens: int = 0

    @property
    def drop(self) -> float | None:
        if self.real is None or self.corrupted is None:
            return None
        return self.real - self.corrupted


async def run_perturbation(
    questions: list,
    rag: Retriever,
    judge: Judge,
    cache: Cache,
    *,
    generator: str,
    gate: float = THRESHOLD,
    pause: float = 15.0,
    k: int = 3,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_score: Callable[[PerturbScore], None] | None = None,
) -> list[PerturbScore]:
    scores: list[PerturbScore] = []
    for q in questions:
        score = PerturbScore(id=q.id)
        scores.append(score)
        hits = rag.retrieve(q.question, k=k)
        entry = cache.answers.get(answer_key(generator, build_prompt(q.question, hits)))
        if entry is None:
            score.error = "no cached answer: run legal_rag.faithfulness first"
        else:
            body = strip_citations(entry["answer"])
            contexts = contexts_of(hits)
            real = cache.scores.get(score_key(judge.name, q.question, body, contexts, True))
            change = corrupt(body)
            if real is None:
                score.error = "no cached real score: run legal_rag.faithfulness first"
            elif body.lstrip().startswith(_REFUSAL):
                score.skipped = "a refusal has no fact to corrupt"
            elif real["faithfulness"] < gate:
                score.skipped = f"real score {real['faithfulness']:.2f} is already below the gate"
            elif change is None:
                score.skipped = "no corruption rule applies to this answer"
            else:
                score.real = real["faithfulness"]
                score.kind, score.change = change.kind, f"{change.before} -> {change.after}"
                key = score_key(judge.name, q.question, change.text, contexts, True)
                if key in cache.scores:
                    score.corrupted = cache.scores[key]["faithfulness"]
                else:
                    try:
                        judged = await judge.score(q.question, change.text, contexts)
                    except Exception as exc:  # reported, never cached, does not stop the run
                        score.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                    else:
                        score.corrupted = judged.value
                        score.live_tokens = judged.prompt_tokens + judged.completion_tokens
                        cache.scores[key] = {
                            "question_id": f"{q.id}~perturbed",
                            "judge": judge.name,
                            "stripped": True,
                            "scored_answer": change.text,
                            "faithfulness": judged.value,
                            "prompt_tokens": judged.prompt_tokens,
                            "completion_tokens": judged.completion_tokens,
                        }
                        cache.save()
                    await sleep(pause)
        if on_score:
            on_score(score)
    return scores


def summarise_perturbation(scores: list[PerturbScore]) -> dict:
    done = [s for s in scores if s.drop is not None]
    detected = [s.id for s in done if s.drop >= MIN_DROP]
    missed = [s.id for s in done if s.drop < MIN_DROP]
    failed = sum(1 for s in scores if s.error)
    rate = len(detected) / len(done) if done else None
    return {
        "questions": len(scores),
        "corrupted": len(done),
        "skipped": sum(1 for s in scores if s.skipped),
        "failed": failed,
        "mean_real": sum(s.real for s in done) / len(done) if done else None,
        "mean_corrupted": sum(s.corrupted for s in done) / len(done) if done else None,
        "detected": detected,
        "missed": missed,
        "detection_rate": rate,
        "sensitive": rate >= MIN_DETECTED if done and not failed else None,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_perturbation(score: PerturbScore) -> str:
    if score.error:
        return f"[{score.id}] ERROR {score.error}\n"
    if score.skipped:
        return f"[{score.id}] skipped: {score.skipped}\n"
    return (
        f"[{score.id}] real={_fmt(score.real)} corrupted={_fmt(score.corrupted)} "
        f"drop={_fmt(score.drop)} ({score.kind}: {score.change})\n"
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
    if not questions:
        sys.stderr.write("no question matches the filters\n")
        return 2

    judge = RagasJudge(args.judge, args.max_tokens, args.reasoning_effort)
    scores = asyncio.run(
        run_perturbation(
            questions,
            load_rag(),
            judge,
            Cache(args.cache),
            generator=settings.groq_model,
            gate=args.threshold,
            pause=args.pause,
            on_score=lambda s: sys.stdout.write(format_perturbation(s)),
        )
    )
    summary = summarise_perturbation(scores)
    sys.stdout.write(
        f"\nquestions: {summary['questions']}  corrupted and scored: {summary['corrupted']}"
        f"  skipped: {summary['skipped']}  failed: {summary['failed']}\n"
        f"mean faithfulness, real answer:       {_fmt(summary['mean_real'])}\n"
        f"mean faithfulness, corrupted answer:  {_fmt(summary['mean_corrupted'])}\n"
        f"noticed (drop >= {MIN_DROP}): {len(summary['detected'])} of {summary['corrupted']}"
        f"  (needed {MIN_DETECTED:.0%})\n"
        f"not noticed: {summary['missed']}\n"
    )
    if summary["sensitive"] is None:
        sys.stdout.write("verdict: not conclusive (some scores failed or nothing was corrupted)\n")
        return 2
    sys.stdout.write(
        "verdict: the metric is SENSITIVE (a corrupted answer scores lower)\n"
        if summary["sensitive"]
        else "verdict: the metric does NOT notice small errors: do not trust it as a fine gate\n"
    )
    return 0 if summary["sensitive"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
