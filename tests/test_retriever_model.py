import json

import numpy as np
import pandas as pd
import pytest

from legal_rag.retriever_model import LegalRetriever, fuse, slug

ARTICLES = [{"article_number": n, "citation": f"Art. {n}"} for n in (10, 20, 30)]
# unit vectors: article 10 is "x", 20 is "y", 30 is the diagonal
X, Y = np.array([1.0, 0.0]), np.array([0.0, 1.0])
D = np.array([1.0, 1.0]) / np.sqrt(2)
AR = np.stack([X, Y, D]).astype(np.float32)
EN = np.stack([Y, X, D]).astype(np.float32)


def model(fusion, matrices, weights=(), query=X, variants=("ar", "en")):
    m = LegalRetriever()
    cfg = {
        "embedding_model": "fake",
        "variants": list(variants),
        "fusion": fusion,
        "weights": list(weights),
    }
    m.setup(cfg, ARTICLES, matrices, encode=lambda _q: query)
    return m


def numbers(m, k=3):
    return [h["article_number"] for h in m.retrieve("q", k)]


def test_slug_matches_the_cache_file_naming():
    assert slug("ar+en") == "ar_en" and slug("ar") == "ar"


def test_single_ranks_by_the_one_matrix():
    assert numbers(model("single", [AR], variants=("ar",))) == [10, 30, 20]


def test_max_keeps_the_best_of_the_two_vectors():
    # query x: article 10 scores 1.0 in ar, article 20 scores 1.0 in en
    assert numbers(model("max", [AR, EN]), k=2) == [10, 20]


def test_weighted_is_a_weighted_sum():
    m = model("weighted", [AR, EN], weights=(0.9, 0.1))
    assert numbers(m) == [10, 30, 20]
    assert m.retrieve("q", 1)[0]["score"] == pytest.approx(0.9)


def test_predict_returns_one_row_per_hit_with_the_columns():
    m = model("single", [AR], variants=("ar",))
    out = m.predict(None, pd.DataFrame({"question": ["a", "b"]}), params={"k": 2})
    assert list(out.columns) == ["question", "rank", "article_number", "citation", "score"]
    assert len(out) == 4 and out["rank"].tolist() == [1, 2, 1, 2]


def test_k_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        model("single", [AR], variants=("ar",)).retrieve("q", 0)


@pytest.mark.parametrize(
    "bad",
    [
        {"fusion": "mean"},
        {"fusion": "weighted", "weights": [1.0]},
    ],
)
def test_setup_validates_the_config(bad):
    cfg = {"embedding_model": "f", "variants": ["ar", "en"], "fusion": "max", "weights": [], **bad}
    with pytest.raises(ValueError):
        LegalRetriever().setup(cfg, ARTICLES, [AR, EN])


def test_matrix_must_have_one_row_per_article():
    with pytest.raises(ValueError):
        model("single", [AR[:2]], variants=("ar",))


def test_unknown_fusion_in_fuse():
    with pytest.raises(ValueError):
        fuse([AR[:, 0]], "median")


def test_load_context_reads_the_artifacts(tmp_path):
    cfg = {"embedding_model": "f", "variants": ["ar+en"], "fusion": "single", "weights": []}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "articles.json").write_text(json.dumps(ARTICLES))
    np.save(tmp_path / "vectors_ar_en.npy", AR)
    arts = {
        "config": str(tmp_path / "config.json"),
        "articles": str(tmp_path / "articles.json"),
        "vectors_ar_en": str(tmp_path / "vectors_ar_en.npy"),
    }
    m = LegalRetriever()
    m.load_context(type("Ctx", (), {"artifacts": arts})())
    m._encode = lambda _q: X
    assert numbers(m) == [10, 30, 20]
