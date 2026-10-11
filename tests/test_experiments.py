import re
from contextlib import contextmanager

import numpy as np
import pytest

from legal_rag.data import Article
from legal_rag.evaluate import EvalQuestion, evaluate
from legal_rag.experiments import (
    ALPHAS,
    MatrixRetriever,
    MemoEmbedder,
    RunContext,
    active_articles,
    embedded_corpus,
    flat_metrics,
    index_cost,
    record,
    run_model,
    sweep_configs,
    variant_text,
)

METRIC_NAME = re.compile(r"^[A-Za-z0-9_\-. /]+$")  # what MLflow accepts


def article(number: int, ar: str = "ar", en: str = "en", repealed: bool = False) -> Article:
    return Article(
        article_number=number,
        text_ar=ar,
        text_en=en,
        is_repealed=repealed,
        citation=f"Art. {number}",
        book="",
        chapter="",
        section="",
        topic="",
        source_page=0,
    )


class LookupEmbedder:
    """Vectors come from dictionaries, so each test controls exactly what is similar to what."""

    def __init__(self, passages: dict[str, list[float]], queries: dict[str, list[float]]) -> None:
        self._passages = passages
        self._queries = queries
        self.passage_calls = 0
        self.query_calls = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.passage_calls += 1
        return [self._passages[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._queries[text]


class LengthEmbedder:
    """Deterministic stand-in that accepts any text."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        vec = np.array([len(text) % 7 + 1, len(text) % 5 + 1], dtype=float)
        return (vec / np.linalg.norm(vec)).tolist()


# ---- what is embedded ----------------------------------------------------------------------


def test_variant_text_covers_arabic_english_and_both():
    a = article(1, ar="نص", en="text")
    assert variant_text(a, "ar") == "نص"
    assert variant_text(a, "en") == "text"
    assert variant_text(a, "ar+en") == "نص\ntext"


def test_a_missing_english_text_falls_back_to_arabic():
    a = article(452, ar="نص", en="")
    assert variant_text(a, "en") == "نص"
    assert variant_text(a, "ar+en") == "نص"


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        variant_text(article(1), "fr")


def test_repealed_and_empty_articles_are_not_indexed():
    kept = active_articles([article(1), article(2, repealed=True), article(3, ar="  ")])
    assert [a.article_number for a in kept] == [1]


# ---- cache ---------------------------------------------------------------------------------


def test_embedded_corpus_is_computed_once_and_cached_on_disk(tmp_path):
    emb = LookupEmbedder({"ar1": [1.0, 0.0], "ar2": [0.0, 1.0]}, {})
    articles = [article(1, ar="ar1"), article(2, ar="ar2")]
    first = embedded_corpus(emb, "org/model", articles, "ar", tmp_path)
    second = embedded_corpus(emb, "org/model", articles, "ar", tmp_path)
    assert emb.passage_calls == 1
    assert np.array_equal(first, second)
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_editing_a_text_invalidates_the_cache(tmp_path):
    emb = LookupEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]}, {})
    embedded_corpus(emb, "m", [article(1, ar="a")], "ar", tmp_path)
    embedded_corpus(emb, "m", [article(1, ar="b")], "ar", tmp_path)
    assert emb.passage_calls == 2
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_each_query_is_embedded_once():
    emb = LookupEmbedder({}, {"q": [1.0, 0.0]})
    memo = MemoEmbedder(emb)
    memo.embed_query("q")
    memo.embed_query("q")
    assert emb.query_calls == 1


# ---- ranking and fusion --------------------------------------------------------------------

ARTICLES = [article(1), article(2)]
# query = [1, 0]. Arabic vectors favour article 1, English vectors favour article 2.
AR = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=np.float32)  # similarity 0.8 and 0.6
EN = np.array([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32)  # similarity 0.6 and 1.0
QUERY = LookupEmbedder({}, {"q": [1.0, 0.0]})
GOLD_IS_2 = [EvalQuestion("q", "en", "q", (2,))]


def top1(fusion: str, matrices: list[np.ndarray], weights: tuple[float, ...] = ()) -> int:
    retriever = MatrixRetriever(QUERY, ARTICLES, matrices, fusion, weights)
    return retriever.retrieve("q", 1)[0].article_number


def test_single_matrix_ranks_by_cosine():
    assert top1("single", [AR]) == 1
    assert top1("single", [EN]) == 2


def test_max_fusion_keeps_the_best_of_the_two_vectors():
    assert top1("max", [AR, EN]) == 2  # article 2 scores 1.0 through its English vector


def test_weighted_fusion_moves_with_the_weight():
    assert top1("weighted", [AR, EN], (0.9, 0.1)) == 1
    assert top1("weighted", [AR, EN], (0.1, 0.9)) == 2


def test_fusion_changes_the_evaluation_result():
    arabic_only = MatrixRetriever(QUERY, ARTICLES, [AR], "single")
    both = MatrixRetriever(QUERY, ARTICLES, [AR, EN], "max")
    assert evaluate(arabic_only, GOLD_IS_2)["overall"]["hit@1"] == 0.0
    assert evaluate(both, GOLD_IS_2)["overall"]["hit@1"] == 1.0


def test_hits_carry_the_article_and_a_cosine_score():
    hit = MatrixRetriever(QUERY, ARTICLES, [AR], "single").retrieve("q", 2)[0]
    assert hit.article_number == 1
    assert hit.citation == "Art. 1"
    assert hit.score == pytest.approx(0.8)


def test_a_matrix_with_the_wrong_number_of_rows_is_rejected():
    with pytest.raises(ValueError, match="one row per"):
        MatrixRetriever(QUERY, ARTICLES, [AR[:1]], "single")


def test_unknown_fusion_is_rejected():
    with pytest.raises(ValueError, match="unknown fusion"):
        MatrixRetriever(QUERY, ARTICLES, [AR], "median").retrieve("q", 1)


# ---- the sweep and what is logged ----------------------------------------------------------


def test_sweep_has_three_families_and_enough_runs():
    configs = sweep_configs()
    assert len(configs) >= 13
    assert {c.family for c in configs} == {"single-vector", "two-vector-max", "two-vector-weighted"}
    assert len({c.name for c in configs}) == len(configs)


def test_weighted_configs_cover_alpha_zero_to_one_and_sum_to_one():
    weighted = [c for c in sweep_configs() if c.fusion == "weighted"]
    assert [c.weights[0] for c in weighted] == list(ALPHAS)
    assert all(sum(c.weights) == pytest.approx(1.0) for c in weighted)


def test_metric_names_are_valid_for_mlflow():
    result = {
        "overall": {"hit@1": 0.5, "hit@3": 0.7, "mrr": 0.6},
        "by_lang": {"ar": {"hit@3": 1.0}, "en": {"hit@3": 0.4}},
        "misses": [],
    }
    metrics = flat_metrics(result)
    assert metrics["hit_at_3"] == 0.7
    assert metrics["en_hit_at_3"] == 0.4
    assert all(METRIC_NAME.match(name) for name in metrics)


class FakeMlflow:
    """Records what would be sent to the tracking server."""

    def __init__(self) -> None:
        self.runs: list[dict] = []
        self._stack: list[dict] = []

    @contextmanager
    def start_run(self, run_name: str, nested: bool = False):
        run = {
            "name": run_name,
            "nested": nested,
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": [],
        }
        self.runs.append(run)
        self._stack.append(run)
        try:
            yield run
        finally:
            self._stack.pop()

    def log_params(self, params: dict) -> None:
        self._stack[-1]["params"].update(params)

    def log_metrics(self, metrics: dict) -> None:
        self._stack[-1]["metrics"].update(metrics)

    def set_tags(self, tags: dict) -> None:
        self._stack[-1]["tags"].update(tags)

    def log_dict(self, data, name: str) -> None:
        self._stack[-1]["artifacts"].append(name)

    def log_text(self, text: str, name: str) -> None:
        self._stack[-1]["artifacts"].append(name)


def corpus() -> list[Article]:
    return [article(n, ar=f"arabic text {n}" * n, en=f"english text {n}" * n) for n in range(1, 9)]


QUESTIONS = [
    EvalQuestion("a-ar", "ar", "سؤال", (1,)),
    EvalQuestion("a-en", "en", "question", (1,)),
]


def test_a_run_logs_params_tags_metrics_and_the_misses(capsys):
    mlflow = FakeMlflow()
    config = sweep_configs()[0]
    result = {
        "overall": {"hit@1": 0.0, "hit@3": 0.5, "hit@5": 1.0, "mrr": 0.4},
        "by_lang": {"ar": {"hit@3": 1.0}},
        "misses": [],
    }
    ctx = RunContext(
        params={"n_docs": 3},
        tags={"git_commit": "abc123", "data_version": "d1", "author": "me", "framework": "f"},
        texts={"requirements.txt": "numpy==2.0"},
    )
    record(mlflow, "run", config, "org/model", result, ctx, {"n_vectors": 3.0, "index_mb": 0.1})
    run = mlflow.runs[0]
    assert run["params"]["embedding_model"] == "org/model"
    assert run["params"]["n_docs"] == 3
    assert run["tags"]["family"] == "single-vector"
    assert run["tags"]["git_commit"] == "abc123"
    assert {"data_version", "author", "framework"} <= set(run["tags"])
    assert run["metrics"]["hit_at_3"] == 0.5
    assert run["metrics"]["n_vectors"] == 3.0
    assert run["artifacts"] == ["misses.json", "requirements.txt"]
    assert "hit@3=0.500" in capsys.readouterr().out


def test_a_dry_run_prints_but_logs_nothing(capsys):
    config = sweep_configs()[0]
    result = {"overall": {"hit@3": 0.5, "mrr": 0.4}, "by_lang": {}, "misses": []}
    record(None, "run", config, "org/model", result, RunContext())
    assert "hit@3=0.500" in capsys.readouterr().out


def test_run_model_logs_every_config_and_a_parent_for_the_sweep(tmp_path):
    mlflow = FakeMlflow()
    run_model(
        mlflow,
        "org/multilingual-e5-small",
        MemoEmbedder(LengthEmbedder()),
        corpus(),
        QUESTIONS,
        tmp_path,
        RunContext(params={"n_docs": 8}),
    )
    nested = [r for r in mlflow.runs if r["nested"]]
    parents = [r for r in mlflow.runs if "weighted sweep" in r["name"]]
    assert len(mlflow.runs) == len(sweep_configs()) + 1  # one extra run: the sweep's parent
    assert len(nested) == len(ALPHAS)
    assert len(parents) == 1
    assert "best_alpha" in parents[0]["params"]
    assert "best_hit_at_3" in parents[0]["metrics"]
    assert len(list(tmp_path.glob("*.npy"))) == 3  # one embedding pass per variant


def test_runs_report_how_big_an_index_each_config_needs(tmp_path):
    mlflow = FakeMlflow()
    run_model(
        mlflow,
        "org/multilingual-e5-small",
        MemoEmbedder(LengthEmbedder()),
        corpus(),
        QUESTIONS,
        tmp_path,
        RunContext(),
    )
    by_name = {r["name"]: r["metrics"] for r in mlflow.runs}
    single, double = by_name["e5-small / ar"], by_name["e5-small / ar and en vectors, max"]
    assert single["n_vectors"] == 8
    assert double["n_vectors"] == 16
    assert double["index_mb"] > single["index_mb"]


def test_index_cost_counts_every_vector_and_megabyte():
    one_megabyte = np.zeros((1000, 250), dtype=np.float32)  # 1000 * 250 * 4 bytes
    cost = index_cost([one_megabyte, one_megabyte])
    assert cost["n_vectors"] == 2000
    assert cost["index_mb"] == pytest.approx(2.0)


def test_run_model_without_mlflow_still_evaluates(tmp_path, capsys):
    run_model(
        None,
        "org/multilingual-e5-small",
        MemoEmbedder(LengthEmbedder()),
        corpus(),
        QUESTIONS,
        tmp_path,
        RunContext(),
    )
    assert capsys.readouterr().out.count("hit@3=") == len(sweep_configs())
