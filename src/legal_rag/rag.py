"""Ingest + query for the Civil Code RAG: one chunk per article -> embed -> Chroma.

Sources returned to callers are article citations, never chunk ids. Repealed articles are
not indexed; use ``LegalRAG.get_article`` to tell the user that an article no longer exists.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.errors import NotFoundError

from legal_rag.config import settings
from legal_rag.data import Article, load_corpus
from legal_rag.logging_conf import configure_logging, correlation_context

log = logging.getLogger(__name__)

COLLECTION = "civil_code"


class IndexNotBuiltError(RuntimeError):
    """The Chroma index does not exist yet: run `python -m legal_rag.rag ingest`."""


class Embedder(Protocol):
    """Anything that turns text into vectors; lets tests swap in a tiny fake."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class E5Embedder:
    """multilingual-e5 via sentence-transformers (e5 needs the query:/passage: prefixes)."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        name = model_name or settings.embedding_model
        log.info("loading embedding model", extra={"model": name})
        self._model = SentenceTransformer(name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            [f"passage: {t}" for t in texts], batch_size=32, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()


@dataclass(frozen=True)
class Hit:
    article_number: int
    citation: str
    text_ar: str
    score: float  # cosine similarity, higher is better
    metadata: dict = field(default_factory=dict)


def _metadata(article: Article) -> dict[str, str | int]:
    return {
        "article_number": article.article_number,
        "citation": article.citation,
        "book": article.book,
        "chapter": article.chapter,
        "section": article.section,
        "topic": article.topic,
        "source_page": article.source_page,
    }


def _client(chroma_dir: Path | None) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(chroma_dir or settings.chroma_dir))


def build_index(
    articles: list[Article],
    embedder: Embedder,
    chroma_dir: Path | None = None,
    batch_size: int = 64,
) -> int:
    """Ingest: one chunk per active article. Replaces the collection, so it is idempotent."""
    client = _client(chroma_dir)
    existing = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    rows = [a for a in articles if not a.is_repealed and a.text_ar.strip()]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        collection.add(
            ids=[str(a.article_number) for a in batch],
            embeddings=embedder.embed_passages([a.text_ar for a in batch]),
            documents=[a.text_ar for a in batch],
            metadatas=[_metadata(a) for a in batch],
        )
    log.info("index built", extra={"indexed": len(rows), "skipped": len(articles) - len(rows)})
    return len(rows)


class LegalRAG:
    """Query side: embed the question, search Chroma, return article-level hits."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        chroma_dir: Path | None = None,
        articles: list[Article] | None = None,
        low_score_threshold: float | None = None,
    ) -> None:
        self.embedder = embedder or E5Embedder()
        self._low_score_threshold = (
            settings.low_score_threshold if low_score_threshold is None else low_score_threshold
        )
        try:
            self._collection = _client(chroma_dir).get_collection(COLLECTION)
        except NotFoundError as exc:
            log.error("vector index not found", extra={"collection": COLLECTION})
            raise IndexNotBuiltError(
                "Chroma index missing: run `python -m legal_rag.rag ingest` first"
            ) from exc
        corpus = articles if articles is not None else load_corpus()
        self._articles = {a.article_number: a for a in corpus}

    @property
    def documents_indexed(self) -> int:
        return self._collection.count()  # used by /health

    def retrieve(self, question: str, k: int = 3) -> list[Hit]:
        if not question.strip():
            log.warning("rejected empty question")  # client mistake, not a server fault
            raise ValueError("question must not be empty")
        start = time.perf_counter()
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(question)],
            n_results=min(k, self.documents_indexed),
        )
        hits = [
            Hit(
                article_number=int(id_),
                citation=meta["citation"],
                text_ar=doc,
                score=1.0 - dist,
                metadata=meta,
            )
            for id_, doc, meta, dist in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        log.debug("question text", extra={"question": question})  # full text only at DEBUG
        log.debug(
            "retrieved",
            extra={
                "articles": [h.article_number for h in hits],
                "scores": [round(h.score, 3) for h in hits],
            },
        )
        log.info(
            "question served",
            extra={
                "k": k,
                "latency_ms": latency_ms,
                "top_article": hits[0].article_number if hits else None,
                "top_score": round(hits[0].score, 3) if hits else None,
                "question_chars": len(question),
            },
        )
        if hits and hits[0].score < self._low_score_threshold:
            log.warning(
                "low retrieval confidence",
                extra={
                    "top_score": round(hits[0].score, 3),
                    "threshold": self._low_score_threshold,
                },
            )
        return hits

    def get_article(self, number: int) -> Article | None:
        """Look up any article, including repealed ones (they are not in the index)."""
        return self._articles.get(number)


def sources_of(hits: list[Hit]) -> list[str]:
    """Article citations for the API's `sources` field (not chunk ids)."""
    return [h.citation for h in hits]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m legal_rag.rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="embed the corpus and (re)build the Chroma index")
    query = sub.add_parser("query", help="retrieve the top-k articles for a question")
    query.add_argument("question")
    query.add_argument("-k", type=int, default=3)
    args = ap.parse_args(argv)
    configure_logging(settings.log_level)
    with correlation_context():  # one id per CLI run; the API sets one per request
        if args.cmd == "ingest":
            build_index(load_corpus(), E5Embedder())
        else:
            for hit in LegalRAG().retrieve(args.question, args.k):
                sys.stdout.write(f"{hit.score:.3f}  {hit.citation}\n")


if __name__ == "__main__":
    main()
