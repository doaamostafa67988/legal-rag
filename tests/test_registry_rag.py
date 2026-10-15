import json
import logging

import numpy as np
import pytest

from legal_rag import registry_rag
from legal_rag.data import Article
from legal_rag.registry_rag import (
    Bundle,
    ModelMismatchError,
    RegistryRAG,
    check_compatible,
    load_bundle,
    log_fields,
    parse_uri,
)
from legal_rag.retriever_model import LegalRetriever
from legal_rag.retriever_model import fuse as model_fuse

X, Y = np.array([1.0, 0.0]), np.array([0.0, 1.0])
D = np.array([1.0, 1.0]) / np.sqrt(2)
AR = np.stack([X, Y, D]).astype(np.float32)
EN = np.stack([Y, X, D]).astype(np.float32)
NUMBERS = (10, 20, 30)


def article(n: int) -> Article:
    return Article(
        article_number=n, text_ar=f"نص {n}", text_en="", is_repealed=False,
        citation=f"Art. {n}", book="", chapter="", section="", topic="", source_page=1,
    )  # fmt: skip


class FixedEmbedder:
    def __init__(self, vector) -> None:
        self.vector = vector

    def embed_passages(self, texts):
        raise NotImplementedError

    def embed_query(self, text):
        return list(self.vector)


def bundle(fusion="max", weights=(), variants=("ar", "en"), matrices=(AR, EN)) -> Bundle:
    config = {
        "embedding_model": "intfloat/multilingual-e5-small",
        "variants": list(variants),
        "fusion": fusion,
        "weights": list(weights),
        "corpus_sha256": "abc",
    }
    rows = [{"article_number": n, "citation": f"Art. {n}"} for n in NUMBERS]
    return Bundle(config, rows, list(matrices))


def rag(b: Bundle, query=X) -> RegistryRAG:
    return RegistryRAG(b, FixedEmbedder(query), articles=[article(n) for n in NUMBERS])


def test_max_fusion_returns_the_best_of_both_vectors():
    hits = rag(bundle("max")).retrieve("q", k=2)
    assert [h.article_number for h in hits] == [10, 20]
    assert hits[0].text_ar == "نص 10" and hits[0].citation == "Art. 10"


def test_single_fusion_uses_one_matrix():
    r = rag(bundle("single", variants=("ar",), matrices=(AR,)))
    assert [h.article_number for h in r.retrieve("q", k=3)] == [10, 30, 20]


def test_weighted_fusion_is_a_weighted_sum():
    hits = rag(bundle("weighted", (0.9, 0.1))).retrieve("q", k=1)
    assert hits[0].article_number == 10 and hits[0].score == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("fusion", "weights"), [("single", ()), ("max", ()), ("weighted", (0.3, 0.7))]
)
def test_serving_ranks_exactly_like_the_registered_pyfunc_model(fusion, weights):
    b = bundle(fusion, weights)
    served = rag(b, D).retrieve("q", k=3)
    model = LegalRetriever()
    model.setup(b.config, b.articles, b.matrices, encode=lambda _q: D)
    registered = model.retrieve("q", k=3)
    assert [h.article_number for h in served] == [h["article_number"] for h in registered]
    assert [h.score for h in served] == pytest.approx([h["score"] for h in registered])


def test_fuse_matches_the_pyfunc_fuse():
    sims = [AR @ D, EN @ D]
    for fusion, weights in (("single", []), ("max", []), ("weighted", [0.2, 0.8])):
        assert np.allclose(
            registry_rag.fuse(sims, fusion, weights), model_fuse(sims, fusion, weights)
        )


def test_k_larger_than_the_index_is_capped():
    assert len(rag(bundle()).retrieve("q", k=10)) == 3


def test_empty_question_is_a_value_error():
    with pytest.raises(ValueError):
        rag(bundle()).retrieve("   ")


def test_documents_indexed_and_get_article():
    r = rag(bundle())
    assert r.documents_indexed == 3 and r.get_article(20).citation == "Art. 20"


def test_articles_missing_from_the_corpus_are_refused():
    with pytest.raises(ModelMismatchError):
        RegistryRAG(bundle(), FixedEmbedder(X), articles=[article(10)])


def test_bad_bundle_shapes_are_refused():
    with pytest.raises(ValueError):
        rag(bundle(matrices=(AR[:2], EN[:2])))
    with pytest.raises(ValueError):
        rag(bundle("weighted", (1.0,)))
    with pytest.raises(ValueError):
        rag(bundle("median"))


def test_load_bundle_finds_files_under_artifacts(tmp_path):
    d = tmp_path / "1" / "artifacts"
    d.mkdir(parents=True)
    b = bundle("single", variants=("ar+en",), matrices=(AR,))
    (d / "config.json").write_text(json.dumps(b.config))
    (d / "articles.json").write_text(json.dumps(b.articles))
    np.save(d / "vectors_ar_en.npy", AR)
    (tmp_path / "1" / "config.json").write_text("{}")  # a decoy: artifacts/ wins
    loaded = load_bundle(tmp_path / "1")
    assert loaded.config["variants"] == ["ar+en"] and loaded.matrices[0].shape == AR.shape


def test_load_bundle_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path)


@pytest.mark.parametrize(
    ("uri", "parsed"),
    [
        ("models:/legal-rag-retriever@production", ("legal-rag-retriever", "production", None)),
        ("models:/legal-rag-retriever/3", ("legal-rag-retriever", None, "3")),
    ],
)
def test_parse_uri(uri, parsed):
    assert parse_uri(uri) == parsed


@pytest.mark.parametrize("uri", ["models:/x", "runs:/abc/model", "legal-rag-retriever@production"])
def test_parse_uri_rejects_other_shapes(uri):
    with pytest.raises(ValueError):
        parse_uri(uri)


def test_check_compatible_accepts_a_matching_build(tmp_path, monkeypatch):
    corpus = tmp_path / "c.json"
    corpus.write_bytes(b"[]")
    b = bundle()
    b.config["corpus_sha256"] = registry_rag._sha256(corpus)
    monkeypatch.setattr(registry_rag.settings, "embedding_model", b.config["embedding_model"])
    check_compatible(b, corpus)


def test_check_compatible_refuses_another_embedding_model(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_rag.settings, "embedding_model", "intfloat/multilingual-e5-base")
    with pytest.raises(ModelMismatchError, match="e5-base"):
        check_compatible(bundle(), tmp_path / "c.json")


def test_check_compatible_refuses_another_corpus(tmp_path, monkeypatch):
    corpus = tmp_path / "c.json"
    corpus.write_bytes(b"[1]")
    model = bundle().config["embedding_model"]
    monkeypatch.setattr(registry_rag.settings, "embedding_model", model)
    with pytest.raises(ModelMismatchError, match="corpus"):
        check_compatible(bundle(), corpus)


def test_log_fields_never_collide_with_logrecord_attributes():
    info = {
        "name": "legal-rag-retriever", "version": "1", "alias": "production", "run_id": "r",
        "embedding_model": "m", "variants": ["ar"], "fusion": "max",
    }  # fmt: skip
    reserved = set(logging.LogRecord("n", logging.INFO, "p", 1, "m", (), None).__dict__)
    assert not reserved & set(log_fields(info))
    # what the service really does; INFO must be enabled or the record is never even built
    logger = logging.getLogger("legal_rag.test_log_fields")
    logger.setLevel(logging.INFO)
    logger.info("loaded", extra=log_fields(info))
    with pytest.raises(KeyError):
        logger.info("loaded", extra=info)  # the bug that stopped the container: name is reserved
