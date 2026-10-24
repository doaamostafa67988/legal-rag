"""The faithfulness negative control, with a fake retriever and a fake judge (no Groq)."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from legal_rag.faithfulness import (
    Cache,
    JudgeResult,
    answer_key,
    contexts_of,
    score_key,
    strip_citations,
)
from legal_rag.faithfulness_control import foreign_pairs, run_control, summarise_control
from legal_rag.generate import build_prompt

GENERATOR = "fake-generator"
JUDGE = "fake-judge"


@dataclass
class Q:
    id: str
    question: str


class FakeRAG:
    def __init__(self, retrieved: dict[str, list[int]]) -> None:
        self.retrieved = retrieved

    def retrieve(self, question: str, k: int = 3) -> list:
        numbers = self.retrieved[question]
        return [SimpleNamespace(article_number=n, text_ar=f"نص {n}") for n in numbers]


class FakeJudge:
    name = JUDGE

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[list[str]] = []

    async def score(self, question: str, response: str, contexts: list[str]) -> JudgeResult:
        self.calls.append(contexts)
        return JudgeResult(self.value, 10, 5)


class FailingJudge(FakeJudge):
    async def score(self, question: str, response: str, contexts: list[str]) -> JudgeResult:
        raise RuntimeError("rate limit")


async def _no_sleep(_seconds: float) -> None:
    return None


QUESTIONS = [Q("q1", "سؤال 1"), Q("q2", "سؤال 2"), Q("q3", "سؤال 3")]
RETRIEVED = {"سؤال 1": [1, 2], "سؤال 2": [3, 4], "سؤال 3": [1, 5]}


def _cache(tmp_path, *, with_answers: bool = True) -> Cache:
    """A cache that already holds each question's answer and its real (own-context) score."""
    cache = Cache(tmp_path / "cache.json")
    if not with_answers:
        return cache
    for q in QUESTIONS:
        hits = FakeRAG(RETRIEVED).retrieve(q.question)
        answer = f"الإجابة عن {q.question} بحسب ما ورد"
        cache.answers[answer_key(GENERATOR, build_prompt(q.question, hits))] = {"answer": answer}
        key = score_key(JUDGE, q.question, strip_citations(answer), contexts_of(hits), True)
        cache.scores[key] = {"faithfulness": 1.0, "prompt_tokens": 1, "completion_tokens": 1}
    return cache


def _run(judge, cache):
    rag = FakeRAG(RETRIEVED)
    return asyncio.run(
        run_control(QUESTIONS, rag, judge, cache, generator=GENERATOR, sleep=_no_sleep)
    )


def test_foreign_pairs_never_share_an_article():
    own = [{1, 2}, {3, 4}, {1, 5}]
    partners = foreign_pairs(own, own)
    assert partners == [1, 2, 1]  # tried in the order i+1, i+2, ...; q3 skips q1 (shares 1)
    assert all(not own[i] & own[j] for i, j in enumerate(partners))


def test_foreign_pairs_is_none_when_everyone_overlaps():
    assert foreign_pairs([{1}, {1}], [{1}, {1}]) == [None, None]


def test_a_judge_that_sees_the_mismatch_makes_the_metric_discriminate(tmp_path):
    judge = FakeJudge(0.0)
    scores = _run(judge, _cache(tmp_path))
    assert [s.control for s in scores] == [0.0, 0.0, 0.0]
    assert [s.real for s in scores] == [1.0, 1.0, 1.0]
    assert judge.calls[0] == ["المادة 3: نص 3", "المادة 4: نص 4"]  # q1 was judged on q2's articles
    assert scores[0].foreign_articles == [3, 4]
    summary = summarise_control(scores)
    assert summary["informative"] is True and summary["mean_control"] == 0.0


def test_a_judge_that_scores_everything_high_is_flagged_blind(tmp_path):
    summary = summarise_control(_run(FakeJudge(0.95), _cache(tmp_path)))
    assert summary["informative"] is False
    assert summary["high"] == ["q1", "q2", "q3"]


def test_a_second_run_is_free_because_the_control_scores_are_cached(tmp_path):
    first, second = FakeJudge(0.0), FakeJudge(0.0)
    _run(first, _cache(tmp_path))
    _run(second, Cache(tmp_path / "cache.json"))  # reload what the first run saved
    assert len(first.calls) == 3 and second.calls == []


def test_a_question_without_a_cached_answer_is_an_error_not_a_crash(tmp_path):
    scores = _run(FakeJudge(0.0), _cache(tmp_path, with_answers=False))
    assert all(s.error and s.control is None for s in scores)
    assert summarise_control(scores)["informative"] is None


def test_a_failing_judge_is_reported_and_not_cached(tmp_path):
    cache = _cache(tmp_path)
    before = len(cache.scores)
    scores = _run(FailingJudge(0.0), cache)
    assert all("RuntimeError" in (s.error or "") for s in scores)
    assert len(cache.scores) == before
