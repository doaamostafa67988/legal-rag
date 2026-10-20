"""The refusal split, with a fake retriever and a pre-filled cache (no Groq, no model)."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from legal_rag.faithfulness import Cache, answer_key, contexts_of, score_key, strip_citations
from legal_rag.faithfulness_refusals import (
    build_rows,
    classify,
    is_refusal,
    summarise_refusals,
)
from legal_rag.generate import build_prompt

GENERATOR = "fake-generator"
JUDGE = "fake-judge"
REFUSAL = "لا تحتوي المواد المرفقة على إجابة لهذا السؤال، حيث لا تتناول المواد (672، 824) الموضوع."


@dataclass
class Q:
    id: str
    question: str
    gold_articles: tuple[int, ...]


class FakeRAG:
    def __init__(self, retrieved: dict[str, list[int]]) -> None:
        self.retrieved = retrieved

    def retrieve(self, question: str, k: int = 3) -> list:
        return [
            SimpleNamespace(article_number=n, text_ar=f"نص {n}") for n in self.retrieved[question]
        ]


QUESTIONS = [
    Q("a", "سؤال a", (44,)),  # answered
    Q("b", "سؤال b", (125,)),  # refused, gold not retrieved: correct
    Q("c", "سؤال c", (418,)),  # refused although gold was retrieved: wrong
]
RETRIEVED = {"سؤال a": [44, 46], "سؤال b": [672, 824], "سؤال c": [418, 434]}
ANSWERS = {"a": ("سن الرشد إحدى وعشرون سنة.", 1.0), "b": (REFUSAL, 0.8), "c": (REFUSAL, 0.0)}


def _cache(tmp_path, *, skip: set[str] = frozenset(), no_score: set[str] = frozenset()) -> Cache:
    cache = Cache(tmp_path / "cache.json")
    rag = FakeRAG(RETRIEVED)
    for q in QUESTIONS:
        if q.id in skip:
            continue
        hits = rag.retrieve(q.question)
        text, value = ANSWERS[q.id]
        cache.answers[answer_key(GENERATOR, build_prompt(q.question, hits))] = {"answer": text}
        if q.id not in no_score:
            key = score_key(JUDGE, q.question, strip_citations(text), contexts_of(hits), True)
            cache.scores[key] = {"faithfulness": value, "prompt_tokens": 1, "completion_tokens": 1}
    return cache


def _rows(cache):
    return build_rows(QUESTIONS, FakeRAG(RETRIEVED), cache, generator=GENERATOR, judge=JUDGE)


@pytest.mark.parametrize(
    "answer",
    [
        REFUSAL,
        "**لا تحتوي المواد المرفقة** على تعريف لعقد البيع.",
        "لا تتناول المواد المرفقة هذا الموضوع.",
    ],
)
def test_refusals_are_recognised(answer):
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "لا يجوز لأحد المتعاقدين تعديل العقد بإرادته المنفردة.",
        "لا، لا يمكن تصحيح العقد الباطل بالإجازة.",
        "سن الرشد إحدى وعشرون سنة، ولا تحتوي المادة 44 على استثناء.",
    ],
)
def test_real_answers_are_not_mistaken_for_refusals(answer):
    assert not is_refusal(answer)


def test_each_question_is_answered_or_refused_right_or_wrong(tmp_path):
    rows = _rows(_cache(tmp_path))
    assert [r.status for r in rows] == ["answered", "refused-correct", "refused-wrong"]


def test_summary_separates_answered_from_refused(tmp_path):
    summary = summarise_refusals(_rows(_cache(tmp_path)))
    assert summary["answered"] == 1 and summary["refused"] == 2
    assert summary["refused_correct"] == 1 and summary["refused_wrong"] == 1
    assert summary["mean_answered"] == pytest.approx(1.0)
    assert summary["mean_refused"] == pytest.approx(0.4)
    assert summary["mean_all"] == pytest.approx(0.6)
    assert summary["refusal_rate"] == pytest.approx(2 / 3)


def test_missing_answers_and_scores_are_reported_not_crashed_on(tmp_path):
    rows = _rows(_cache(tmp_path, skip={"a"}, no_score={"b"}))
    assert [r.status for r in rows][:2] == ["missing", "missing"]
    assert "run legal_rag.faithfulness first" in rows[0].note + rows[1].note
    summary = summarise_refusals(rows)
    assert summary["missing"] == 2 and summary["questions"] == 3


def test_classify_reads_the_article_numbers_the_answer_cites(tmp_path):
    cache = _cache(tmp_path)
    q = Q("a", "سؤال a", (44,))
    row = classify(
        q, FakeRAG(RETRIEVED).retrieve(q.question), cache, generator=GENERATOR, judge=JUDGE
    )
    assert row.retrieved == (44, 46) and row.gold == (44,)
