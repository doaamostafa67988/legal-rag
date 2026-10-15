"""The sensitivity check, with a fake retriever and a fake judge (no Groq)."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from legal_rag.faithfulness import (
    Cache,
    JudgeResult,
    answer_key,
    contexts_of,
    score_key,
    strip_citations,
)
from legal_rag.faithfulness_perturb import corrupt, run_perturbation, summarise_perturbation
from legal_rag.generate import build_prompt

GENERATOR = "fake-generator"
JUDGE = "fake-judge"


@dataclass
class Q:
    id: str
    question: str


class FakeRAG:
    def retrieve(self, question: str, k: int = 3) -> list:
        return [SimpleNamespace(article_number=1, text_ar="نص 1")]


class FakeJudge:
    name = JUDGE

    def __init__(self, score_of) -> None:
        self.score_of = score_of
        self.calls: list[str] = []

    async def score(self, question: str, response: str, contexts: list[str]) -> JudgeResult:
        self.calls.append(response)
        return JudgeResult(self.score_of(response), 10, 5)


class FailingJudge(FakeJudge):
    async def score(self, question: str, response: str, contexts: list[str]) -> JudgeResult:
        raise RuntimeError("rate limit")


async def _no_sleep(_seconds: float) -> None:
    return None


# question -> (cached answer, real score)
ANSWERS = {
    "q1": ("لا يجوز لأحد المتعاقدين تعديل العقد بإرادته المنفردة.", 1.0),  # ruling flip
    "q2": ("مدة التقادم هي ثلاث سنوات من يوم العلم بالضرر.", 1.0),  # time limit
    "q3": ("سعر الفائدة القانونية 4% في المسائل المدنية.", 1.0),  # figure
    "q4": ("لا تحتوي المواد المرفقة على إجابة لهذا السؤال.", 0.5),  # refusal: skipped
    "q5": ("العقد شريعة المتعاقدين في كل الأحوال.", 0.4),  # already below the gate: skipped
    "q6": ("العقد شريعة المتعاقدين في كل الأحوال تقريبا.", 1.0),  # no rule applies: skipped
}
QUESTIONS = [Q(qid, f"سؤال {qid}") for qid in ANSWERS]
ORIGINALS = {strip_citations(text) for text, _ in ANSWERS.values()}


def _cache(tmp_path, *, with_answers: bool = True) -> Cache:
    cache = Cache(tmp_path / "cache.json")
    if not with_answers:
        return cache
    hits = FakeRAG().retrieve("")
    for q in QUESTIONS:
        text, real = ANSWERS[q.id]
        cache.answers[answer_key(GENERATOR, build_prompt(q.question, hits))] = {"answer": text}
        key = score_key(JUDGE, q.question, strip_citations(text), contexts_of(hits), True)
        cache.scores[key] = {"faithfulness": real, "prompt_tokens": 1, "completion_tokens": 1}
    return cache


def _run(judge, cache):
    return asyncio.run(
        run_perturbation(QUESTIONS, FakeRAG(), judge, cache, generator=GENERATOR, sleep=_no_sleep)
    )


def _notices(response: str) -> float:
    return 1.0 if response in ORIGINALS else 0.4  # a judge that sees every corruption


def test_a_figure_is_changed_but_article_numbers_and_paragraphs_are_not():
    found = corrupt("نصت المادة 163 (الفقرة 2) على مدة 5 سنوات")
    assert found.kind == "figure"
    assert "المادة 163" in found.text and "الفقرة 2" in found.text
    assert "11 سنوات" in found.text


def test_list_markers_are_not_treated_as_facts():
    found = corrupt("1. العقد شريعة المتعاقدين.\n2. الفائدة 4% سنويا.")
    assert found.text.startswith("1. العقد") and "9%" in found.text


def test_time_limits_in_words_and_the_ruling_are_corrupted():
    assert "خمس سنوات" in corrupt("التقادم بمضي ثلاث سنوات").text
    assert corrupt("لا يجوز تعديل العقد").text == "يجوز تعديل العقد"
    assert corrupt("يجوز للقاضي منح أجل").text == "لا يجوز للقاضي منح أجل"


def test_list_markers_in_the_middle_of_a_line_are_not_figures():
    joined = "بالترتيب التالي: 1. بمقتضى العرف. 2. فإذا لم يوجد. الفائدة 4% سنويا."
    found = corrupt(joined)  # strip_citations joins the lines, so the markers are mid-line
    assert "1. بمقتضى" in found.text and "2. فإذا" in found.text and "9%" in found.text
    assert corrupt("(1) العقد شريعة المتعاقدين و (2) غيره") is None


def test_a_yes_no_particle_is_not_a_fact_to_corrupt():
    assert corrupt("نعم، يجب تنفيذ العقد بطريق تتفق مع حسن النية.") is None
    assert corrupt("لا، لا يمكن تصحيح العقد الباطل.") is None


def test_nothing_to_corrupt_gives_none():
    assert corrupt("العقد شريعة المتعاقدين") is None


def test_only_good_non_refusal_answers_with_a_rule_are_judged(tmp_path):
    judge = FakeJudge(_notices)
    scores = _run(judge, _cache(tmp_path))
    assert len(judge.calls) == 3  # q4 refusal, q5 below the gate and q6 no rule are skipped
    assert [s.skipped is not None for s in scores] == [False, False, False, True, True, True]
    assert [s.kind for s in scores[:3]] == ["ruling", "time limit or rate", "figure"]


def test_a_judge_that_sees_the_errors_makes_the_metric_sensitive(tmp_path):
    summary = summarise_perturbation(_run(FakeJudge(_notices), _cache(tmp_path)))
    assert summary["sensitive"] is True and summary["detection_rate"] == 1.0
    assert summary["mean_real"] == pytest.approx(1.0)
    assert summary["mean_corrupted"] == pytest.approx(0.4)


def test_a_judge_that_ignores_the_errors_is_flagged(tmp_path):
    summary = summarise_perturbation(_run(FakeJudge(lambda _r: 1.0), _cache(tmp_path)))
    assert summary["sensitive"] is False
    assert summary["missed"] == ["q1", "q2", "q3"]


def test_a_second_run_is_free_because_the_corrupted_scores_are_cached(tmp_path):
    first, second = FakeJudge(_notices), FakeJudge(_notices)
    _run(first, _cache(tmp_path))
    _run(second, Cache(tmp_path / "cache.json"))  # reload what the first run saved
    assert len(first.calls) == 3 and second.calls == []


def test_a_missing_answer_is_an_error_not_a_crash(tmp_path):
    scores = _run(FakeJudge(_notices), _cache(tmp_path, with_answers=False))
    assert all(s.error for s in scores)
    assert summarise_perturbation(scores)["sensitive"] is None


def test_a_failing_judge_is_reported_and_not_cached(tmp_path):
    cache = _cache(tmp_path)
    before = len(cache.scores)
    scores = _run(FailingJudge(_notices), cache)
    assert sum("RuntimeError" in (s.error or "") for s in scores) == 3
    assert len(cache.scores) == before
