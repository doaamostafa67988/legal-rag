"""Tests for ingest + query (fixtures live in conftest.py)."""

import pytest

from legal_rag.data import load_corpus
from legal_rag.rag import LegalRAG, build_index, sources_of


def test_ingest_skips_repealed_and_empty_articles(rag):
    assert rag.documents_indexed == 3


def test_ingest_is_idempotent(articles, embedder, tmp_path):
    build_index(articles, embedder, chroma_dir=tmp_path)
    build_index(articles, embedder, chroma_dir=tmp_path)
    rag = LegalRAG(embedder=embedder, chroma_dir=tmp_path, articles=articles)
    assert rag.documents_indexed == 3


def test_retrieve_puts_the_matching_article_first(rag):
    hits = rag.retrieve("ما هو سن الرشد", k=3)
    assert hits[0].article_number == 44
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_retrieve_respects_k_and_caps_at_index_size(rag):
    assert len(rag.retrieve("العقد", k=1)) == 1
    assert len(rag.retrieve("العقد", k=50)) == 3


def test_empty_question_is_rejected(rag):
    with pytest.raises(ValueError):
        rag.retrieve("   ")


def test_sources_are_article_citations_not_chunk_ids(rag):
    hits = rag.retrieve("من ارتكب خطأ سبب ضررا", k=1)  # shares exact tokens with art. 163
    assert sources_of(hits) == ["Egyptian Civil Code, Article 163"]


def test_repealed_article_is_looked_up_but_never_retrieved(rag):
    assert rag.get_article(60).is_repealed
    assert 60 not in {h.article_number for h in rag.retrieve("ملغاة", k=3)}


def test_full_corpus_index_excludes_exactly_the_56_repealed(embedder, tmp_path):
    corpus = load_corpus()
    assert build_index(corpus, embedder, chroma_dir=tmp_path) == 1149 - 56
    rag = LegalRAG(embedder=embedder, chroma_dir=tmp_path, articles=corpus)
    assert rag.documents_indexed == 1093
    assert rag.get_article(147).topic == "آثار العقد"
