"""RAGAS faithfulness of the served answers, with a disk cache so nothing is paid for twice.

    uv run python -m legal_rag.faithfulness --lang ar --n 15
    uv run python -m legal_rag.faithfulness --ids q13-ar --debug
    uv run python -m legal_rag.faithfulness --threshold 0.75     # exit code 1 below it: a CI gate

For every question it retrieves the articles, generates the answer with the service's own
generator and asks a *different* Groq model (the judge) how much of the answer the articles support.

Two caches live in one JSON file (data/eval/faithfulness_cache.json):
  * answers, keyed by (generator, system prompt, user prompt): the prompt holds the question and the
    retrieved articles, so the same retrieval gives the same cached answer without a Groq call;
  * scores, keyed by (judge, question, answer, contexts, how the answer was prepared).
Anything that changes the pipeline's output changes a key, so the score is recomputed for real.

The judge scores the answer without its citation boilerplate ("... وفقاً للمادة 418"): those
clauses are not legal claims and made single scores swing. Which articles the answer cites, and
whether they were among the retrieved ones, is checked separately and for free.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from legal_rag.config import ROOT, settings
from legal_rag.evaluate import EvalQuestion, Retriever, load_questions, load_rag
from legal_rag.generate import SYSTEM, ChatClient, GroqChat, answer, build_prompt

DEFAULT_CACHE = ROOT / "data" / "eval" / "faithfulness_cache.json"
DEFAULT_JUDGE = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
THRESHOLD = 0.75
CACHE_VERSION = 1
PREP_VERSION = 2  # bump when strip_citations changes: old scores no longer apply

# ---- citations ------------------------------------------------------------------------------

_DIGITS = "0-9٠-٩"
_NOUN = r"(?:ماد(?:ة|ت|تين|تان)|مواد)"
_ART_PREFIX = r"(?:في\s+ال|في\s+|لل|بال|ال|ل|ب)?"
_NUMBERS = rf"[{_DIGITS}]+(?:\s*[،,]?\s*و?\s*[{_DIGITS}]+)*"
_B = r"\*{0,3}"  # markdown bold may wrap a reference or a label: "**المادة 1**"
# "الفقرة 1", "، فقرة (2)", "(الفقرة 1)" and "(الفقرة الأولى)": digits or an ordinal word
_ORDINAL = r"(?:ال)?(?:أولى|اولى|ثانية|ثالثة|رابعة|خامسة)"
_SUBPART = rf"(?:\s*[،,]?\s*\(?\s*(?:ال)?فقر[ةت]\s*(?:\(?\s*[{_DIGITS}]+\s*\)?|{_ORDINAL})\s*\)?)?"
_ARTICLE_REF = (
    rf"{_B}(?:(?:ل|ب|في\s+)?نص\s+)?{_ART_PREFIX}{_NOUN}\s*(?:رقم\s*)?{_NUMBERS}{_B}{_SUBPART}"
)
_LEAD = (
    r"(?:و?بناء[ًا]*\s+على|و?استناد[اً]*\s+(?:إلى|الى|على)|و?وفق[اً]*|و?طبق[اً]*"
    r"|و?حسب|و?بحسب|و?بموجب|كما\s+(?:ورد|جاء|نص|نصت)\s+في)"
)
# "المادة المعتمدة: المادة 148 (الفقرة 1)." / "الاعتماد: المادة 172." are labels, not claims
_LABEL_WORDS = (
    r"(?:(?:ال)?ماد[ةت]\s+(?:ال)?معتمدة|(?:ال)?مواد\s+(?:ال)?معتمدة"
    r"|الاعتماد|المرجع|المراجع|السند|المصدر)"
)
_LABEL = re.compile(rf"{_B}{_LABEL_WORDS}{_B}\s*:{_B}\s*{_ARTICLE_REF}\s*[.،]?{_B}")
# "اعتمدت في إجابتي على المادة 163 التي تنص صراحةً على أن:" -> only the quoted text stays
_SELF_CITE = re.compile(
    rf"اعتمدت\s+في\s+(?:إجابتي|اجابتي)\s+على\s*{_ARTICLE_REF}"
    r"(?:\s*(?:التي|الذي)\s+(?:تنص|ينص|نصت)\s*(?:صراحة[ًا]*)?\s*(?:على\s+أن)?\s*:?)?"
)
# a connective is required: "تنص المادة 44 على ..." is a claim and must stay
_CITE_CLAUSE = re.compile(rf"[،,]?\s*(?:وذلك\s+)?{_LEAD}\s*{_ARTICLE_REF}\s*[،,]?")
_ATTACHED = re.compile(rf"{_LEAD}\s*{_ART_PREFIX}مواد\s+(?:ال)?مرفقة\s*[،,:]?")
_PARENTHETICAL = re.compile(rf"\(\s*{_ARTICLE_REF}\s*\)")
_CITED = re.compile(rf"{_NOUN}\s*(?:رقم\s*)?({_NUMBERS})")
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def strip_citations(text: str) -> str:
    """Remove citation boilerplate; fall back to the raw text if nothing is left.

    Removed: "بناءً على المادة 44" clauses, "(المادة 418)", label lines such as
    "المادة المعتمدة: المادة 148" and "اعتمدت في إجابتي على المادة 163 التي تنص على أن:".
    Kept: claims such as "فقد نصت المادة 141 على أن ...".
    """
    out = _SELF_CITE.sub(" ", text)
    out = _LABEL.sub(" ", out)
    out = _ATTACHED.sub(" ", out)
    out = _CITE_CLAUSE.sub(" ", out)
    out = _PARENTHETICAL.sub(" ", out)
    out = re.sub(r"\s+([.،,؟?!])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = out.strip(" ،,:")
    return out if len(out) >= 10 else text


def cited_articles(text: str) -> list[int]:
    found: list[int] = []
    for group in _CITED.findall(text.translate(_DIGIT_MAP)):
        found += [int(n) for n in re.findall(r"\d+", group)]
    return sorted(set(found))


# ---- cache ----------------------------------------------------------------------------------


def _key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


def answer_key(generator: str, user_prompt: str) -> str:
    return _key("answer", generator, SYSTEM, user_prompt)


def score_key(judge: str, question: str, response: str, contexts: list[str], strip: bool) -> str:
    return _key("score", CACHE_VERSION, PREP_VERSION, judge, question, response, contexts, strip)


class Cache:
    """Answers and scores in one JSON file, saved after every live call (a crash loses nothing)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.answers: dict[str, dict] = {}
        self.scores: dict[str, dict] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                data = {}
            if data.get("version") == CACHE_VERSION:
                self.answers = data.get("answers", {})
                self.scores = data.get("scores", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_VERSION, "answers": self.answers, "scores": self.scores}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)  # atomic: a half-written cache is never seen


# ---- the judge ------------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeResult:
    value: float
    prompt_tokens: int
    completion_tokens: int


class Judge(Protocol):
    name: str

    async def score(self, question: str, response: str, contexts: list[str]) -> JudgeResult: ...


class RagasJudge:
    """RAGAS Faithfulness through Groq's OpenAI-compatible endpoint, counting every token."""

    def __init__(
        self,
        model: str = DEFAULT_JUDGE,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> None:
        import httpx
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness

        if settings.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY is not set: put it in .env")
        self.name = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.http_errors = 0
        self.last_outputs: list[str] = []
        self._outputs: list[str] = []
        http = httpx.AsyncClient(event_hooks={"response": [self._count]}, timeout=120.0)
        client = AsyncOpenAI(
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=GROQ_BASE_URL,
            http_client=http,
            max_retries=2,
        )
        kwargs: dict = {"max_tokens": max_tokens}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        self._metric = Faithfulness(llm=llm_factory(model, client=client, **kwargs))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    async def _count(self, response) -> None:
        await response.aread()
        if response.status_code >= 400:
            self.http_errors += 1
            return
        try:
            body = response.json()
        except ValueError:
            return
        if not isinstance(body, dict):
            return
        usage = body.get("usage") or {}
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        message = ((body.get("choices") or [{}])[0] or {}).get("message") or {}
        calls = message.get("tool_calls") or []
        if calls:
            self._outputs.append(str(calls[0].get("function", {}).get("arguments", "")))
        else:
            self._outputs.append(str(message.get("content") or ""))

    async def score(self, question: str, response: str, contexts: list[str]) -> JudgeResult:
        before = (self.prompt_tokens, self.completion_tokens, len(self._outputs))
        result = await self._metric.ascore(
            user_input=question, response=response, retrieved_contexts=contexts
        )
        self.last_outputs = self._outputs[before[2] :]
        return JudgeResult(
            float(result.value), self.prompt_tokens - before[0], self.completion_tokens - before[1]
        )


# ---- scoring --------------------------------------------------------------------------------


@dataclass
class QuestionScore:
    id: str
    lang: str
    faithfulness: float | None = None
    error: str | None = None
    answer: str = ""
    retrieved: list[int] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    answer_cached: bool = False
    score_cached: bool = False
    live_tokens: int = 0  # judge tokens spent on this call (0 when the score came from the cache)
    cold_tokens: int = 0  # what this score costs without the cache

    @property
    def citation_valid(self) -> bool:
        """The answer cites at least one article and every cited article was retrieved."""
        return bool(self.cited) and set(self.cited) <= set(self.retrieved)


def contexts_of(hits: list) -> list[str]:
    """The judge sees what the generator saw: "المادة 147: <text>"."""
    return [f"المادة {h.article_number}: {h.text_ar}" for h in hits]


async def score_question(
    q: EvalQuestion,
    rag: Retriever,
    chat: ChatClient,
    judge: Judge,
    cache: Cache,
    *,
    generator: str,
    strip: bool = True,
    refresh_answers: bool = False,
    k: int = 3,
) -> QuestionScore:
    hits = rag.retrieve(q.question, k=k)
    result = QuestionScore(id=q.id, lang=q.lang, retrieved=[h.article_number for h in hits])
    prompt = build_prompt(q.question, hits)
    akey = answer_key(generator, prompt)
    if akey in cache.answers and not refresh_answers:
        result.answer = cache.answers[akey]["answer"]
        result.answer_cached = True
    else:
        result.answer = await asyncio.to_thread(answer, q.question, hits, chat)
        cache.answers[akey] = {"generator": generator, "question_id": q.id, "answer": result.answer}
        cache.save()
    result.cited = cited_articles(result.answer)

    body = strip_citations(result.answer) if strip else result.answer
    contexts = contexts_of(hits)
    skey = score_key(judge.name, q.question, body, contexts, strip)
    if skey in cache.scores:
        hit = cache.scores[skey]
        result.faithfulness = hit["faithfulness"]
        result.cold_tokens = hit["prompt_tokens"] + hit["completion_tokens"]
        result.score_cached = True
        return result
    try:
        judged = await judge.score(q.question, body, contexts)
    except Exception as exc:  # a failed score is reported, never cached, and does not stop the run
        result.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        return result
    result.faithfulness = judged.value
    result.live_tokens = result.cold_tokens = judged.prompt_tokens + judged.completion_tokens
    cache.scores[skey] = {
        "question_id": q.id,
        "judge": judge.name,
        "stripped": strip,
        "scored_answer": body,
        "faithfulness": judged.value,
        "prompt_tokens": judged.prompt_tokens,
        "completion_tokens": judged.completion_tokens,
    }
    cache.save()
    return result


async def score_all(
    questions: list[EvalQuestion],
    rag: Retriever,
    chat: ChatClient,
    judge: Judge,
    cache: Cache,
    *,
    generator: str,
    strip: bool = True,
    refresh_answers: bool = False,
    pause: float = 15.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_score: Callable[[QuestionScore], None] | None = None,
) -> list[QuestionScore]:
    """Score every question; the pause only follows calls that really reached Groq."""
    scores: list[QuestionScore] = []
    for q in questions:
        score = await score_question(
            q, rag, chat, judge, cache, generator=generator, strip=strip,
            refresh_answers=refresh_answers,
        )  # fmt: skip
        scores.append(score)
        if on_score:
            on_score(score)
        if not (score.answer_cached and score.score_cached):
            await sleep(pause)
    return scores


def summarise(scores: list[QuestionScore], threshold: float = THRESHOLD) -> dict:
    done = [(s.id, s.faithfulness) for s in scores if s.faithfulness is not None]
    n = len(scores)
    return {
        "n": n,
        "scored": len(done),
        "failed": n - len(done),
        "mean_faithfulness": sum(v for _, v in done) / len(done) if done else None,
        "below_threshold": [qid for qid, v in done if v < threshold],
        "citation_rate": sum(bool(s.cited) for s in scores) / n if n else 0.0,
        "citation_valid_rate": sum(s.citation_valid for s in scores) / n if n else 0.0,
        "answers_from_cache": sum(s.answer_cached for s in scores),
        "scores_from_cache": sum(s.score_cached for s in scores),
        "live_tokens": sum(s.live_tokens for s in scores),
        "cold_tokens": sum(s.cold_tokens for s in scores),
    }


def passes(summary: dict, threshold: float = THRESHOLD) -> bool:
    """The gate: every question scored, and the mean at or above the threshold."""
    mean = summary["mean_faithfulness"]
    return summary["failed"] == 0 and mean is not None and mean >= threshold


def format_summary(summary: dict, threshold: float = THRESHOLD) -> str:
    mean = summary["mean_faithfulness"]
    lines = [
        f"questions: {summary['n']}  scored: {summary['scored']}  failed: {summary['failed']}",
        f"mean faithfulness: {'n/a' if mean is None else f'{mean:.3f}'}"
        f"  (gate {threshold}: {'PASS' if passes(summary, threshold) else 'FAIL'})",
        f"below {threshold}: {summary['below_threshold']}",
        f"answers citing an article: {summary['citation_rate']:.2f}"
        f"  citing only retrieved articles: {summary['citation_valid_rate']:.2f}",
        f"from cache: {summary['answers_from_cache']} answers, "
        f"{summary['scores_from_cache']} scores",
        f"judge tokens spent now: {summary['live_tokens']:,}"
        f"  (the same run without the cache: {summary['cold_tokens']:,})",
    ]
    return "\n".join(lines) + "\n"


def format_score(score: QuestionScore) -> str:
    source = "cache" if score.score_cached else "judge"
    value = "FAILED" if score.faithfulness is None else f"{score.faithfulness:.2f}"
    extra = f" {score.error}" if score.error else ""
    return (
        f"[{score.id}] faithfulness={value} ({source}) cited={score.cited} "
        f"retrieved={score.retrieved} tokens={score.live_tokens}{extra}\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m legal_rag.faithfulness")
    ap.add_argument("--lang", default=None, help="only questions of this language, e.g. ar")
    ap.add_argument("--ids", nargs="+", default=None, help="only these question ids")
    ap.add_argument("--n", type=int, default=None, help="at most this many questions")
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--max-tokens", type=int, default=4096, help="output cap per judge call")
    ap.add_argument("--reasoning-effort", default=None, help="low | medium | high (gpt-oss)")
    ap.add_argument("--pause", type=float, default=15.0, help="seconds after each live score")
    ap.add_argument("--no-strip", action="store_true", help="score the answer with its citations")
    ap.add_argument("--refresh-answers", action="store_true", help="ignore cached answers")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--debug", action="store_true", help="print the judge's output for live scores")
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

    cache = Cache(args.cache)
    judge = RagasJudge(args.judge, args.max_tokens, args.reasoning_effort)

    def show(score: QuestionScore) -> None:
        sys.stdout.write(format_score(score))
        if args.debug and not score.score_cached:
            for n, out in enumerate(judge.last_outputs, start=1):
                sys.stdout.write(f"  --- judge output {n} ---\n{out[:2000]}\n")

    scores = asyncio.run(
        score_all(
            questions, load_rag(), GroqChat(), judge, cache,
            generator=settings.groq_model, strip=not args.no_strip,
            refresh_answers=args.refresh_answers, pause=args.pause, on_score=show,
        )
    )  # fmt: skip
    summary = summarise(scores, args.threshold)
    sys.stdout.write("\n" + format_summary(summary, args.threshold))
    sys.stdout.write(f"judge tokens this run, failed calls included: {judge.total_tokens:,}\n")
    return 0 if passes(summary, args.threshold) else 1


if __name__ == "__main__":
    sys.exit(main())
