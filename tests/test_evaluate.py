import json

import pytest

import legal_rag.evaluate as ev
from legal_rag.config import settings
from legal_rag.data import load_corpus
from legal_rag.evaluate import EvalQuestion, evaluate, load_questions
from legal_rag.rag import Hit


class ScriptedRag:
    """Returns a fixed ranking of article numbers for each question text."""

    def __init__(self, rankings: dict[str, list[int]]) -> None:
        self._rankings = rankings

    def retrieve(self, question: str, k: int = 3) -> list[Hit]:
        return [
            Hit(article_number=n, citation=f"Art. {n}", text_ar="", score=1.0)
            for n in self._rankings[question][:k]
        ]


QUESTIONS = [
    EvalQuestion("q1", "ar", "first", (10,)),
    EvalQuestion("q2", "en", "second", (20,)),
    EvalQuestion("q3", "ar", "third", (30,)),
    EvalQuestion("q4", "en", "fourth", (40,)),
]
RANKINGS = {
    "first": [10, 5, 6],  # gold at rank 1
    "second": [5, 6, 20],  # gold at rank 3
    "third": [5, 6, 7, 8, 9],  # never found
    "fourth": [5, 6, 7, 8, 40],  # gold at rank 5
}


def test_hit_at_k_and_mrr_are_computed_from_the_gold_rank():
    result = evaluate(ScriptedRag(RANKINGS), QUESTIONS)
    assert result["n"] == 4
    assert result["overall"]["hit@1"] == pytest.approx(1 / 4)
    assert result["overall"]["hit@3"] == pytest.approx(2 / 4)
    assert result["overall"]["hit@5"] == pytest.approx(3 / 4)
    assert result["overall"]["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 5) / 4)


def test_scores_are_split_by_language():
    by_lang = evaluate(ScriptedRag(RANKINGS), QUESTIONS)["by_lang"]
    assert by_lang["ar"]["hit@3"] == pytest.approx(1 / 2)
    assert by_lang["en"]["hit@3"] == pytest.approx(1 / 2)
    assert by_lang["en"]["hit@5"] == pytest.approx(1.0)
    assert by_lang["ar"]["mrr"] == pytest.approx(1 / 2)


def test_misses_list_questions_whose_gold_is_outside_the_top_three():
    misses = evaluate(ScriptedRag(RANKINGS), QUESTIONS)["misses"]
    assert [m["id"] for m in misses] == ["q3", "q4"]
    assert misses[0]["retrieved"] == [5, 6, 7]
    assert misses[0]["gold"] == [30]


def test_any_gold_article_counts_as_a_hit():
    question = EvalQuestion("q", "ar", "text", (7, 8))
    result = evaluate(ScriptedRag({"text": [8, 1, 2]}), [question])
    assert result["overall"]["hit@1"] == 1.0


def test_several_chunks_of_one_article_count_once():
    question = EvalQuestion("q", "ar", "text", (10,))
    # without de-duplication article 10 would sit at rank 4 and be a miss at 3
    result = evaluate(ScriptedRag({"text": [5, 5, 5, 10]}), [question])
    assert result["overall"]["hit@3"] == 1.0


def test_the_headline_cut_off_must_be_requested():
    with pytest.raises(ValueError, match="3"):
        evaluate(ScriptedRag(RANKINGS), QUESTIONS, ks=(1, 5))


def test_shipped_question_set_is_usable():
    questions = load_questions()
    corpus = {a.article_number: a for a in load_corpus()}
    assert len(questions) >= 20
    assert len({q.id for q in questions}) == len(questions)
    assert {q.lang for q in questions} == {"ar", "en"}
    for q in questions:
        for number in q.gold_articles:
            assert number in corpus, q.id
            assert not corpus[number].is_repealed, q.id


def test_arabic_and_english_versions_share_the_gold_article():
    by_id = {q.id: q for q in load_questions()}
    for qid, q in by_id.items():
        if qid.endswith("-ar"):
            assert by_id[qid[:-3] + "-en"].gold_articles == q.gold_articles


def _perfect_rag() -> ScriptedRag:
    return ScriptedRag({q.question: [q.gold_articles[0]] for q in load_questions()})


def test_main_prints_a_readable_report(monkeypatch, capsys):
    monkeypatch.setattr(ev, "load_rag", lambda backend=None: _perfect_rag())
    ev.main([])
    out = capsys.readouterr().out
    assert "hit@3=1.000" in out
    assert "misses" not in out


def test_main_can_print_json(monkeypatch, capsys):
    monkeypatch.setattr(ev, "load_rag", lambda backend=None: _perfect_rag())
    ev.main(["--json"])
    assert json.loads(capsys.readouterr().out)["overall"]["mrr"] == 1.0


def test_load_rag_uses_the_configured_backend(monkeypatch):
    built = {}

    class FakeRag:
        def __init__(self, embedder=None):
            built["embedder"] = embedder

    monkeypatch.setattr(ev, "LegalRAG", FakeRag)
    monkeypatch.setattr(settings, "embedder_backend", "torch")
    ev.load_rag()
    assert built["embedder"] is None  # torch backend: LegalRAG builds its own E5Embedder
