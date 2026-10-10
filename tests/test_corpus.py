"""Validation tests for data/civil_code.json (run: pytest -q)."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "civil_code.json"
REQUIRED = {
    "article_number", "book", "chapter", "section", "topic", "text_ar",
    "text_en", "is_repealed", "source_page", "citation",
}  # fmt: skip


@pytest.fixture(scope="module")
def arts():
    return json.loads(DATA.read_text(encoding="utf-8"))


def by_number(arts, n):
    return next(a for a in arts if a["article_number"] == n)


def test_schema_matches_handbook(arts):
    for a in arts:
        assert REQUIRED <= a.keys()
        assert isinstance(a["article_number"], int)
        assert a["citation"] == f"Egyptian Civil Code, Article {a['article_number']}"


def test_all_1149_articles_contiguous(arts):
    assert [a["article_number"] for a in arts] == list(range(1, 1150))


def test_repealed_ranges(arts):
    repealed = {a["article_number"] for a in arts if a["is_repealed"]}
    assert repealed == set(range(54, 81)) | set(range(389, 418))


def test_no_empty_text_for_active_articles(arts):
    for a in arts:
        if not a["is_repealed"]:
            n = a["article_number"]
            assert a["text_ar"].strip(), f"empty Arabic text: {n}"
            assert a["text_en"].strip() or a["en_missing"], f"empty English text: {n}"


def test_known_source_anomalies_are_flagged(arts):
    assert [a["article_number"] for a in arts if a["en_missing"]] == [452]
    assert [a["article_number"] for a in arts if a["ar_incomplete"]] == [1022]


def test_lam_alef_ligature_fixed(arts):
    assert by_number(arts, 2)["text_ar"].startswith("لا يجوز")
    stray = r"(?<![\u0621-\u064A])ال يجوز"  # "ال" alone before "يجوز" = swapped ligature
    assert not any(re.search(stray, a["text_ar"]) for a in arts)


def test_no_header_fragments_leaked(arts):
    for a in arts:
        assert not re.search(r"\sمادة\s*$", a["text_ar"]), a["article_number"]
        assert not re.search(r"\s\d{1,4}\s*$", a["text_ar"]), a["article_number"]


def test_handbook_example_article_147(arts):
    a = by_number(arts, 147)  # the handbook's own schema example
    assert (a["chapter"], a["section"], a["topic"]) == ("مصادر الالتزام", "العقد", "آثار العقد")
    assert a["source_page"] == 16
    assert "العقد شريعة المتعاقدين" in a["text_ar"]


def test_volume_sizes_match_the_code(arts):
    sizes = Counter(a["volume"] for a in arts if a["volume"])
    assert sorted(sizes.values()) == [120, 228, 329, 384]  # articles 89-1149 in four volumes


def test_article_id_kept_for_the_baseline_notebook(arts):
    assert all(a["article_id"] == str(a["article_number"]) for a in arts)
