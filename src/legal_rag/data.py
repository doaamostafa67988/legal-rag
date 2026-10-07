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
    topic: str
    source_page: int
    volume: str = ""  # the Arabic "الكتاب" level, an extra on top of the handbook schema
    en_missing: bool = False  # English text absent in the source PDF (article 452)
    ar_incomplete: bool = False  # Arabic text incomplete in the source PDF (article 1022)


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
            topic=r["topic"],
            source_page=r["source_page"],
            volume=r["volume"],
            en_missing=r["en_missing"],
            ar_incomplete=r["ar_incomplete"],
        )
        for r in records
    ]
