import json
from dataclasses import dataclass
from pathlib import Path

from legal_rag.config import settings


@dataclass(frozen=True)
class Article:
    article_number: int
    text_ar: str
    text_en: str
    is_repealed: bool
    citation: str
    book: str
    chapter: str
    section: str


def load_corpus(path: Path | None = None) -> list[Article]:
    """Load the parsed Civil Code JSON into a list of Article objects."""
    path = path or settings.corpus_json
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [
        Article(
            article_number=r["article_number"],
            text_ar=r["text_ar"],
            text_en=r["text_en"],
            is_repealed=r["is_repealed"],
            citation=r["citation"],
            book=r["book"],
            chapter=r["chapter"],
            section=r["section"],
        )
        for r in records
    ]
