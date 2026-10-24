"""Check the corpus before anything is trained on it (the "validate data" step of CT).

    uv run python -m legal_rag.validate_data

Fails (exit 1) on a corpus that would quietly poison a candidate: too few articles, repeated
article numbers, an active article with no Arabic text, or with no English text without the
`en_missing` flag the parser sets when the source PDF really lacks it. Repealed articles may be
empty: they are excluded from the index.
"""

import sys
from collections import Counter
from pathlib import Path

from legal_rag.config import settings
from legal_rag.data import Article, load_corpus

MIN_ARTICLES = 1000  # the corpus has 1149 (1093 active); a parser regression shows up as a drop


def validate(articles: list[Article], min_articles: int = MIN_ARTICLES) -> list[str]:
    """Every problem found, as a sentence; an empty list means the corpus is fine."""
    problems: list[str] = []
    if len(articles) < min_articles:
        problems.append(f"only {len(articles)} articles, expected at least {min_articles}")
    repeated = sorted(n for n, c in Counter(a.article_number for a in articles).items() if c > 1)
    if repeated:
        problems.append(f"article numbers appear more than once: {repeated[:10]}")
    active = [a for a in articles if not a.is_repealed]
    no_ar = [a.article_number for a in active if not a.text_ar.strip()]
    if no_ar:
        problems.append(f"{len(no_ar)} active articles have no Arabic text: {no_ar[:10]}")
    no_en = [a.article_number for a in active if not a.text_en.strip() and not a.en_missing]
    if no_en:
        problems.append(f"{len(no_en)} active articles have no English text: {no_en[:10]}")
    no_citation = [a.article_number for a in articles if not a.citation.strip()]
    if no_citation:
        problems.append(f"{len(no_citation)} articles have no citation: {no_citation[:10]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    corpus = Path(argv[0]) if argv else settings.corpus_json
    if not corpus.is_file():
        sys.stderr.write(f"corpus not found: {corpus}\n")
        return 1
    articles = load_corpus(corpus)
    problems = validate(articles)
    active = sum(1 for a in articles if not a.is_repealed)
    sys.stdout.write(f"corpus: {len(articles)} articles, {active} active\n")
    for problem in problems:
        sys.stdout.write(f"PROBLEM: {problem}\n")
    sys.stdout.write("data validation: FAILED\n" if problems else "data validation: ok\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
