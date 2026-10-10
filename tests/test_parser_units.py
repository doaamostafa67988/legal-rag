"""Unit tests for the pure helpers in parse_civil_code (no PDF needed)."""

from legal_rag.parse_civil_code import HDR_EN, _accepted_headers, fix_ar, fix_ligatures, order_lines


def ch(c, width):
    return {"c": c, "bbox": (0.0, 0.0, width, 1.0)}


def test_fix_ar_reverses_digit_runs_and_markers():
    assert fix_ar("المادة ٣١") == "المادة 13"
    assert fix_ar("سنة ٦٥٩١") == "سنة 1956"
    assert fix_ar(")١( نص") == "(1) نص"


def test_fix_ligatures_swaps_only_zero_width_alef():
    assert fix_ligatures([ch("ا", 0.0), ch("ل", 6.0)]) == "لا"  # swapped ligature
    assert fix_ligatures([ch("ا", 2.3), ch("ل", 2.3)]) == "ال"  # normal definite article


def test_header_regex_accepts_missing_space_but_not_ranges():
    assert HDR_EN.match("Article1022").group(1) == "1022"
    assert HDR_EN.match("Articles 54-80 have been repealed") is None


def line(text, x0, x1, y=100.0, col="en"):
    return {"col": col, "y": y, "x0": x0, "x1": x1, "raw": text, "text": text, "bold": False}


def test_order_lines_glues_split_words_in_visual_order():
    parts = [line("f an agreement", 100.2, 180.0), line("In the absence o", 36.0, 100.0)]
    assert order_lines(parts)[0]["text"] == "In the absence of an agreement"


def test_headers_follow_strict_sequence():
    lines = [line("Article 444.", 36, 80, 10), line("Article 452", 36, 80, 50)]
    assert [h[1] for h in _accepted_headers(lines, expected=452)] == [452]
