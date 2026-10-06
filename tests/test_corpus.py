"""Validation tests for data/civil_code.json (run: uv run pytest -q)."""

import json
import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "civil_code.json"


@pytest.fixture(scope="module")
def arts():
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_all_1149_articles_contiguous(arts):
    assert [a["article_number"] for a in arts] == list(range(1, 1150))


def test_repealed_ranges(arts):
    repealed = {a["article_number"] for a in arts if a["is_repealed"]}
    assert repealed == set(range(54, 81)) | set(range(389, 418))


def test_no_empty_text_for_active_articles(arts):
    for a in arts:
        if not a["is_repealed"]:
            assert a["text_ar"].strip(), f"empty Arabic text: {a['article_id']}"
            assert a["text_en"].strip() or a["en_missing"], f"empty English text: {a['article_id']}"


def test_lam_alef_ligature_fixed(arts):
    art2 = next(a for a in arts if a["article_number"] == 2)
    assert art2["text_ar"].startswith("لا يجوز")
    assert not any(re.search(r"(?<![\u0621-\u064A])ال يجوز", a["text_ar"]) for a in arts)


def test_no_header_fragments_leaked(arts):
    for a in arts:
        assert not re.search(r"\sمادة\s*$", a["text_ar"]), a["article_id"]
        assert not re.search(r"\s\d{1,4}\s*$", a["text_ar"]), a["article_id"]


def test_digit_runs_not_reversed(arts):
    # Article 54 in the source cites "Articles 54 to 80" (reversed in the PDF text layer)
    assert next(a for a in arts if a["article_number"] == 1)["citation"].endswith("Art. 1")
