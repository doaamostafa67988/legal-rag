"""Shared fixtures: a tiny hashing embedder stands in for e5, so tests need no model download."""

import math
import re
import zlib

import pytest

from legal_rag.data import Article
from legal_rag.rag import LegalRAG, build_index

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
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


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
def index_dir(articles, embedder, tmp_path):
    build_index(articles, embedder, chroma_dir=tmp_path)
    return tmp_path


@pytest.fixture
def rag(articles, embedder, index_dir) -> LegalRAG:
    return LegalRAG(embedder=embedder, chroma_dir=index_dir, articles=articles)
