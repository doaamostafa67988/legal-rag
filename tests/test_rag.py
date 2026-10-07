"""Tests for ingest + query. A tiny hashing embedder stands in for e5 so no model download."""

import math
import re
import zlib

import pytest

from legal_rag.data import Article, load_corpus
from legal_rag.rag import LegalRAG, build_index, sources_of

DIM = 256


class FakeEmbedder:
    """Bag-of-words hashing into 256 dims: similar wording gives similar vectors."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(tok.encode()) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make(number: int, text_ar: str, repealed: bool = False) -> Article:
    return Article(
        article_number=number,
        text_ar=text_ar,
        text_en="",
        is_repealed=repealed,
        citation=f"Egyptian Civil Code, Article {number}",
        book="",
        chapter="",
        section="",
        topic="",
        source_page=1,
    )


@pytest.fixture
def articles() -> list[Article]:
    return [
        make(44, "سن الرشد احدى وعشرون سنة ميلادية كاملة"),
        make(147, "العقد شريعة المتعاقدين فلا يجوز نقضه ولا تعديله"),
        make(163, "كل خطأ سبب ضررا للغير يلزم من ارتكبه بالتعويض"),
        make(60, "ملغاة", repealed=True),
        make(61, "   "),
    ]


@pytest.fixture
def rag(articles, tmp_path) -> LegalRAG:
    build_index(articles, FakeEmbedder(), chroma_dir=tmp_path)
    return LegalRAG(embedder=FakeEmbedder(), chroma_dir=tmp_path, articles=articles)


def test_ingest_skips_repealed_and_empty_articles(rag):
    assert rag.documents_indexed == 3


def test_ingest_is_idempotent(articles, tmp_path):
    build_index(articles, FakeEmbedder(), chroma_dir=tmp_path)
    build_index(articles, FakeEmbedder(), chroma_dir=tmp_path)
    rag = LegalRAG(embedder=FakeEmbedder(), chroma_dir=tmp_path, articles=articles)
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
    hits = rag.retrieve("الخطأ والتعويض", k=1)
    assert sources_of(hits) == ["Egyptian Civil Code, Article 163"]


def test_repealed_article_is_looked_up_but_never_retrieved(rag):
    assert rag.get_article(60).is_repealed
    assert 60 not in {h.article_number for h in rag.retrieve("ملغاة", k=3)}


def test_full_corpus_index_excludes_exactly_the_56_repealed(tmp_path):
    corpus = load_corpus()
    assert build_index(corpus, FakeEmbedder(), chroma_dir=tmp_path) == 1149 - 56
    rag = LegalRAG(embedder=FakeEmbedder(), chroma_dir=tmp_path, articles=corpus)
    assert rag.documents_indexed == 1093
    assert rag.get_article(147).topic == "آثار العقد"
