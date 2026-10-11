import asyncio
import json

from legal_rag.evaluate import EvalQuestion
from legal_rag.faithfulness import (
    Cache,
    JudgeResult,
    QuestionScore,
    answer_key,
    cited_articles,
    contexts_of,
    format_summary,
    passes,
    score_all,
    score_key,
    strip_citations,
    summarise,
)
from legal_rag.rag import Hit


def hit(number: int) -> Hit:
    return Hit(article_number=number, citation=f"Art. {number}", text_ar=f"نص {number}", score=1.0)


class FakeRag:
    def __init__(self, retrieved: dict[str, list[int]]) -> None:
        self.retrieved = retrieved

    def retrieve(self, question: str, k: int = 3) -> list[Hit]:
        return [hit(n) for n in self.retrieved[question]][:k]


class FakeChat:
    def __init__(self, reply: str = "بناءً على المادة 418، يلتزم البائع بنقل الملكية.") -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


class FakeJudge:
    name = "judge-x"

    def __init__(self, fail_on: tuple[str, ...] = (), value: float = 1.0) -> None:
        self.fail_on = fail_on
        self.value = value
        self.seen: list[tuple[str, str]] = []

    async def score(self, question: str, response: str, contexts: list[str]) -> JudgeResult:
        self.seen.append((question, response))
        if question in self.fail_on:
            raise RuntimeError("rate limited")
        return JudgeResult(self.value, 1000, 500)


QUESTIONS = [
    EvalQuestion("q1", "ar", "سؤال 1", (418,)),
    EvalQuestion("q2", "ar", "سؤال 2", (44,)),
]
RETRIEVED = {"سؤال 1": [418, 481, 466], "سؤال 2": [44, 46, 621]}


def run(cache, judge, chat, rag=None, **kwargs):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    scores = asyncio.run(
        score_all(
            QUESTIONS,
            rag or FakeRag(RETRIEVED),
            chat,
            judge,
            cache,
            generator="gen-x",
            sleep=fake_sleep,
            **kwargs,
        )
    )
    return scores, sleeps


# ---- citations -------------------------------------------------------------------------------


def test_citation_clauses_are_removed_from_the_answer():
    assert (
        strip_citations(
            "بناءً على المواد المرفقة، يُعرَّف عقد البيع بأنه عقد ينقل الملكية، وذلك وفقاً للمادة 418."
        )
        == "يُعرَّف عقد البيع بأنه عقد ينقل الملكية."
    )
    assert (
        strip_citations("بناءً على المادة 44، فقرة (2)، فإن سن الرشد هي إحدى وعشرون سنة.")
        == "فإن سن الرشد هي إحدى وعشرون سنة."
    )
    assert strip_citations("يلتزم البائع بنقل الملكية (المادة 418).") == (
        "يلتزم البائع بنقل الملكية."
    )
    assert strip_citations("وفقاً للمادتين 418 و481 يلتزم المشتري بدفع الثمن.") == (
        "يلتزم المشتري بدفع الثمن."
    )


def test_a_claim_about_what_an_article_says_is_kept():
    claim = "تنص المادة 44 على أن سن الرشد إحدى وعشرون سنة."
    assert strip_citations(claim) == claim
    assert strip_citations("المادة 418 تعرّف البيع بأنه عقد.") == "المادة 418 تعرّف البيع بأنه عقد."


def test_an_answer_that_is_only_a_citation_is_left_alone():
    assert strip_citations("وفقاً للمادة 418.") == "وفقاً للمادة 418."


def test_cited_articles_are_found_with_either_kind_of_digit():
    assert cited_articles("بناءً على المادة ٤٤ وكذلك المادة 46") == [44, 46]
    assert cited_articles("وفقاً للمادتين 418 و481") == [418, 481]
    assert cited_articles("لا تحتوي المواد المرفقة على إجابة") == []


def test_the_judge_sees_the_articles_as_the_generator_did():
    assert contexts_of([hit(147)]) == ["المادة 147: نص 147"]


# ---- cache keys and file ----------------------------------------------------------------------


def test_every_input_that_matters_changes_the_score_key():
    base = score_key("j", "q", "a", ["c"], True)
    assert base == score_key("j", "q", "a", ["c"], True)
    assert base != score_key("other-judge", "q", "a", ["c"], True)
    assert base != score_key("j", "other question", "a", ["c"], True)
    assert base != score_key("j", "q", "other answer", ["c"], True)
    assert base != score_key("j", "q", "a", ["other context"], True)
    assert base != score_key("j", "q", "a", ["c"], False)
    assert answer_key("gen", "prompt") != answer_key("other-gen", "prompt")
    assert answer_key("gen", "prompt") != answer_key("gen", "other prompt")


def test_cache_survives_a_round_trip_and_ignores_a_damaged_file(tmp_path):
    path = tmp_path / "cache.json"
    cache = Cache(path)
    cache.scores["k"] = {"faithfulness": 0.5, "prompt_tokens": 1, "completion_tokens": 2}
    cache.save()
    assert Cache(path).scores["k"]["faithfulness"] == 0.5
    path.write_text("{not json", encoding="utf-8")
    assert Cache(path).scores == {}
    path.write_text(json.dumps({"version": 999, "scores": {"k": {}}}), encoding="utf-8")
    assert Cache(path).scores == {}


# ---- scoring ----------------------------------------------------------------------------------


def test_a_second_run_with_nothing_changed_costs_no_calls(tmp_path):
    path = tmp_path / "cache.json"
    judge, chat = FakeJudge(), FakeChat()
    first, _ = run(Cache(path), judge, chat)
    assert len(judge.seen) == 2
    assert chat.calls == 2
    assert [s.faithfulness for s in first] == [1.0, 1.0]

    second, sleeps = run(Cache(path), judge, chat)  # a fresh Cache: reads the file
    assert len(judge.seen) == 2
    assert chat.calls == 2
    assert [s.faithfulness for s in second] == [1.0, 1.0]
    assert all(s.answer_cached and s.score_cached for s in second)
    assert sleeps == []  # nothing reached Groq, so there is nothing to wait for


def test_the_pause_follows_calls_that_really_reached_groq(tmp_path):
    _, sleeps = run(Cache(tmp_path / "cache.json"), FakeJudge(), FakeChat())
    assert len(sleeps) == len(QUESTIONS)


def test_a_different_retrieval_is_generated_and_scored_again(tmp_path):
    path = tmp_path / "cache.json"
    judge, chat = FakeJudge(), FakeChat()
    run(Cache(path), judge, chat)
    changed = FakeRag({"سؤال 1": [481, 418, 466], "سؤال 2": [44, 46, 621]})
    scores, _ = run(Cache(path), judge, chat, rag=changed)
    assert [s.answer_cached for s in scores] == [False, True]
    assert [s.score_cached for s in scores] == [False, True]
    assert chat.calls == 3
    assert len(judge.seen) == 3


def test_refreshing_answers_regenerates_but_reuses_the_score_of_an_identical_answer(tmp_path):
    path = tmp_path / "cache.json"
    judge, chat = FakeJudge(), FakeChat()
    run(Cache(path), judge, chat)
    scores, _ = run(Cache(path), judge, chat, refresh_answers=True)
    assert chat.calls == 4
    assert len(judge.seen) == 2  # same answers: the judge is not asked again
    assert all(s.score_cached and not s.answer_cached for s in scores)


def test_the_judge_scores_the_answer_without_its_citation_clause(tmp_path):
    judge = FakeJudge()
    run(Cache(tmp_path / "a.json"), judge, FakeChat())
    assert judge.seen[0][1] == "يلتزم البائع بنقل الملكية."
    raw_judge = FakeJudge()
    run(Cache(tmp_path / "b.json"), raw_judge, FakeChat(), strip=False)
    assert raw_judge.seen[0][1].startswith("بناءً على المادة 418")


def test_a_failed_score_is_reported_not_cached_and_does_not_stop_the_run(tmp_path):
    path = tmp_path / "cache.json"
    scores, _ = run(Cache(path), FakeJudge(fail_on=("سؤال 1",)), FakeChat())
    assert scores[0].faithfulness is None
    assert "rate limited" in scores[0].error
    assert scores[1].faithfulness == 1.0
    assert len(Cache(path).scores) == 1
    retry, _ = run(Cache(path), FakeJudge(), FakeChat())
    assert retry[0].faithfulness == 1.0  # scored now: the failure left nothing behind
    assert retry[0].score_cached is False


def test_citations_are_checked_against_the_retrieved_articles(tmp_path):
    scores, _ = run(Cache(tmp_path / "cache.json"), FakeJudge(), FakeChat())
    assert scores[0].cited == [418] and scores[0].citation_valid  # 418 was retrieved
    assert scores[1].cited == [418] and not scores[1].citation_valid  # 418 was not


def test_tokens_are_counted_when_spent_and_when_cached(tmp_path):
    path = tmp_path / "cache.json"
    first, _ = run(Cache(path), FakeJudge(), FakeChat())
    assert sum(s.live_tokens for s in first) == 3000
    second, _ = run(Cache(path), FakeJudge(), FakeChat())
    summary = summarise(second)
    assert summary["live_tokens"] == 0
    assert summary["cold_tokens"] == 3000


# ---- summary and gate -------------------------------------------------------------------------


def _score(qid: str, value: float | None, cited=(418,), retrieved=(418,)) -> QuestionScore:
    return QuestionScore(
        id=qid, lang="ar", faithfulness=value, cited=list(cited), retrieved=list(retrieved)
    )


def test_summary_reports_the_mean_the_low_scores_and_the_citation_rates():
    summary = summarise([_score("a", 1.0), _score("b", 0.5), _score("c", 0.9, cited=(7,))])
    assert summary["mean_faithfulness"] == (1.0 + 0.5 + 0.9) / 3
    assert summary["below_threshold"] == ["b"]
    assert summary["citation_rate"] == 1.0
    assert summary["citation_valid_rate"] == 2 / 3


def test_the_gate_needs_every_question_scored_and_a_high_enough_mean():
    assert passes(summarise([_score("a", 0.9), _score("b", 0.8)]))
    assert not passes(summarise([_score("a", 0.9), _score("b", 0.5)]))
    assert not passes(summarise([_score("a", 1.0), _score("b", None)]))  # unverified is not a pass
    assert not passes(summarise([]))


def test_the_summary_text_says_pass_or_fail():
    assert "PASS" in format_summary(summarise([_score("a", 0.9)]))
    assert "FAIL" in format_summary(summarise([_score("a", 0.1)]))
