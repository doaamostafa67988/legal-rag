"""The finalists comparison: planning is pure, and the hold-out is refused. No Groq, no model."""

from types import SimpleNamespace

import pytest

from legal_rag import faithfulness as faith
from legal_rag.finalists import (
    FINALISTS,
    average_score_tokens,
    finalist_list,
    format_plan,
    main,
    pending_work,
)
from legal_rag.generate import build_prompt

GENERATOR = "fake-generator"
JUDGE = "fake-judge"
QUESTIONS = [SimpleNamespace(id=f"q{i}", question=f"سؤال {i}") for i in (1, 2, 3)]


def _hits(*numbers: int) -> list:
    return [SimpleNamespace(article_number=n, text_ar=f"نص {n}") for n in numbers]


def _cache(tmp_path, *, answered: int, scored: int) -> faith.Cache:
    """Answers for the first `answered` questions, scores for the first `scored` of those."""
    cache = faith.Cache(tmp_path / "cache.json")
    for i, q in enumerate(QUESTIONS[:answered]):
        hits = _hits(i + 1)
        answer = f"جواب {i + 1}"
        cache.answers[faith.answer_key(GENERATOR, build_prompt(q.question, hits))] = {
            "answer": answer
        }
        if i < scored:
            key = faith.score_key(
                JUDGE, q.question, faith.strip_citations(answer), faith.contexts_of(hits), True
            )
            cache.scores[key] = {
                "faithfulness": 1.0,
                "prompt_tokens": 1000,
                "completion_tokens": 400,
            }
    return cache


HITS = [_hits(1), _hits(2), _hits(3)]


def test_everything_is_pending_with_an_empty_cache(tmp_path):
    cache = _cache(tmp_path, answered=0, scored=0)
    answers, scores = pending_work(QUESTIONS, HITS, cache, JUDGE, GENERATOR)
    assert len(answers) == 3 and len(scores) == 3


def test_cached_answers_and_scores_are_not_pending(tmp_path):
    cache = _cache(tmp_path, answered=2, scored=1)
    answers, scores = pending_work(QUESTIONS, HITS, cache, JUDGE, GENERATOR)
    assert len(answers) == 1  # only q3 has no answer
    assert len(scores) == 2  # q2 has an answer but no score, q3 has neither


def test_a_fully_cached_finalist_costs_nothing(tmp_path):
    cache = _cache(tmp_path, answered=3, scored=3)
    assert pending_work(QUESTIONS, HITS, cache, JUDGE, GENERATOR) == (set(), set())


def test_finalists_that_retrieve_the_same_articles_share_their_work(tmp_path):
    cache = _cache(tmp_path, answered=0, scored=0)
    first = pending_work(QUESTIONS, HITS, cache, JUDGE, GENERATOR)
    second = pending_work(QUESTIONS, HITS, cache, JUDGE, GENERATOR)
    assert first[0] | second[0] == first[0] and first[1] | second[1] == first[1]


def test_a_different_retrieval_is_new_work(tmp_path):
    cache = _cache(tmp_path, answered=3, scored=3)
    other = [_hits(1), _hits(2), _hits(30)]  # q3 now retrieves a different article
    answers, scores = pending_work(QUESTIONS, other, cache, JUDGE, GENERATOR)
    assert len(answers) == 1 and len(scores) == 1


def test_average_score_tokens_uses_the_cache_or_a_default(tmp_path):
    assert average_score_tokens(_cache(tmp_path, answered=2, scored=2)) == 1400
    assert average_score_tokens(_cache(tmp_path, answered=0, scored=0)) == 3400


def test_the_plan_states_that_nothing_was_sent():
    text = format_plan([("e5-small / ar", 3, 3)], (3, 3), 1400)
    assert "e5-small / ar" in text and "about 4,200" in text and "nothing was sent to Groq" in text


def test_every_finalist_names_a_config_that_exists():
    assert [f.config.name for f in finalist_list()] == [name for _, name in FINALISTS]


def test_the_holdout_file_is_refused_before_anything_loads():
    with pytest.raises(SystemExit) as exc:
        main(["--questions", "data/eval/holdout.json"])
    assert exc.value.code == 2


def test_pick_outside_the_list_is_refused():
    with pytest.raises(SystemExit):
        main(["--pick", "99"])
