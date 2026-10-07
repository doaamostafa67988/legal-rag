"""Parse the bilingual Egyptian Civil Code PDF into one JSON record per article.

Facts about laws.pdf (all verified against the page images):
- Two columns: English on the left, Arabic on the right, one table row per article.
- The English header "Article N" is the reliable anchor. Arabic digit runs come out
  reversed from the text layer (13 -> 31, 1956 -> 6591), so they are reversed back.
- Word stores the lam-alef ligature reversed ("la" -> "al"); the swapped alef has a
  zero-width bbox, which tells it apart from the normal definite article.
- Hierarchy comes from the Arabic headings (Part/Volume/Chapter/Section/Topic).
- Some English headers are missing in the source (e.g. 452): recovered from the Arabic.
- Repeal notices ("Articles 54-80 ... repealed") become stub records.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pymupdf

from legal_rag.config import settings

log = logging.getLogger(__name__)

ALEFS = set("اأإآ")
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
TO_ASCII = str.maketrans(AR_DIGITS, "0123456789")
HDR_EN = re.compile(r"^Article\s*(\d+)\b\s*(.*)$")
REPEAL = re.compile(r"^\*?\s*Articles?\s+(\d+)\s*(?:-|to)\s*(\d+)\b.*repealed", re.I)
HDR_AR_JUNK = re.compile(r"^[\(\)\s\d٠-٩]*(?:مادة)?[\(\)\s\d٠-٩]*$")
# Handbook schema: book / chapter / section / topic. "volume" (الكتاب) is an extra level.
AR_LEVELS = [
    ("book", re.compile(r"^القسم\s+\S+$")),
    ("volume", re.compile(r"^الكتاب\s+\S+$")),
    ("chapter", re.compile(r"^الباب\s+\S+$")),
    ("section", re.compile(r"^الفصل\s+\S+$")),
]
LEVEL_NAMES = ["book", "volume", "chapter", "section", "topic"]
NUMBERED = re.compile(r"^[٠-٩0-9]+\s*[-–]\s*(.+)$")
ORDINAL = re.compile(r"^(?:أولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)")


def fix_ligatures(chars: list[dict]) -> str:
    """Put swapped lam-alef ligatures back in order (swapped alef has zero width)."""
    out = [c["c"] for c in chars]
    for i in range(len(chars) - 1):
        a, n = chars[i], chars[i + 1]
        if a["c"] in ALEFS and n["c"] == "ل" and (a["bbox"][2] - a["bbox"][0]) < 0.3:
            out[i], out[i + 1] = "ل", a["c"]
    return "".join(out)


def page_lines(page):
    """Yield dicts: col, y, x0, x1, raw, text, bold (majority of chars in a bold font)."""
    mid = page.rect.width / 2
    for b in page.get_text("rawdict")["blocks"]:
        for line in b.get("lines", []):
            chars = [c for s in line["spans"] for c in s["chars"]]
            x0, y0, x1, _ = line["bbox"]
            col = "en" if (x0 + x1) / 2 < mid else "ar"
            if col == "ar":  # stream order can misplace punctuation; use visual RTL order
                chars.sort(key=lambda c: -(c["bbox"][0] + c["bbox"][2]) / 2)
            raw = fix_ligatures(chars)
            text = raw.strip()
            if not text:
                continue
            total = sum(1 for c in chars if c["c"].strip())
            bold = sum(
                1
                for s in line["spans"]
                if "Bold" in s["font"]
                for c in s["chars"]
                if c["c"].strip()
            )
            yield {
                "col": col,
                "y": y0,
                "x0": x0,
                "x1": x1,
                "raw": raw,
                "text": text,
                "bold": total > 0 and bold / total >= 0.5,
            }


def order_lines(lines: list[dict]) -> list[dict]:
    """Sort by y; glue fragments of one visual line (same column, |dy| < 2.5)."""
    lines = sorted(lines, key=lambda d: (d["y"], d["col"] != "en"))
    out, i = [], 0
    while i < len(lines):
        j = i + 1
        while (
            j < len(lines)
            and lines[j]["col"] == lines[i]["col"]
            and abs(lines[j]["y"] - lines[j - 1]["y"]) < 2.5
        ):
            j += 1
        run = sorted(lines[i:j], key=lambda d: d["x0"] if d["col"] == "en" else -d["x0"])
        merged = run[0]
        for nxt in run[1:]:
            gap = nxt["x0"] - merged["x1"] if nxt["col"] == "en" else merged["x0"] - nxt["x1"]
            spaced = merged["raw"].endswith(" ") or nxt["raw"].startswith(" ")
            sep = "" if (gap < 1.5 and not spaced) else " "
            merged = {
                **merged,
                "raw": merged["raw"] + nxt["raw"],
                "text": merged["text"] + sep + nxt["text"],
                "x0": min(merged["x0"], nxt["x0"]),
                "x1": max(merged["x1"], nxt["x1"]),
                "bold": merged["bold"] and nxt["bold"],
            }
        out.append(merged)
        i = j
    return out


def new_article(n: int, page: int, hier: dict, repealed: bool = False) -> dict:
    return {
        "article_number": n,
        "page": page,
        "ar": [],
        "en": [],
        "gap": False,
        "repealed": repealed,
        **hier,
    }


def _accepted_headers(lines: list[dict], expected: int) -> list[tuple]:
    """English 'Article N' headers on a page, accepted only in strict sequence."""
    headers, exp = [], expected
    for ln in lines:
        if ln["col"] != "en":
            continue
        if REPEAL.match(ln["text"]):
            exp = max(exp, int(REPEAL.match(ln["text"]).group(2)) + 1)
            continue
        m = HDR_EN.match(ln["text"])
        if m and int(m.group(1)) in (exp, exp + 1):
            n = int(m.group(1))
            headers.append((ln["y"], n, m.group(2)))
            exp = n + 1
    return headers


def parse(pdf_path: str | Path) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    arts, cur, expected = [], None, 1
    hier = {"book": "", "volume": "", "chapter": "باب تمهيدي", "section": "", "topic": ""}
    pending = None  # hierarchy level waiting for its title line
    for pno, page in enumerate(doc):
        lines = order_lines(page_lines(page))
        headers = _accepted_headers(lines, expected)
        band = [h[0] for h in headers]
        # Arabic lines on the same rows as English bold lines are headings (Arabic body
        # text can be bold too, so we match by row, not by font).
        heading_ys = [
            ln["y"]
            for ln in lines
            if ln["col"] == "en"
            and ln["bold"]
            and not HDR_EN.match(ln["text"])
            and not REPEAL.match(ln["text"])
        ]
        hi = 0
        for ln in lines:
            y, text, col = ln["y"], ln["text"], ln["col"]
            m_hdr = HDR_EN.match(text) if col == "en" else None
            if (
                m_hdr
                and hi < len(headers)
                and abs(headers[hi][0] - y) < 1.0
                and int(m_hdr.group(1)) == headers[hi][1]
            ):
                _, n, rest = headers[hi]
                hi += 1
                if n == expected + 1:  # English header missing for `expected`
                    gap = new_article(expected, pno + 1, hier)
                    gap["gap"] = True
                    arts.append(gap)
                cur = new_article(n, pno + 1, hier)
                if rest:
                    cur["en"].append(rest)
                arts.append(cur)
                expected = n + 1
                continue
            m_rep = REPEAL.match(text) if col == "en" else None
            if m_rep:
                first, last = int(m_rep.group(1)), int(m_rep.group(2))
                for k in range(first, last + 1):
                    if cur is not None and cur["article_number"] == k:
                        cur["repealed"] = True
                    elif k >= expected:
                        arts.append(new_article(k, pno + 1, hier, repealed=True))
                expected = max(expected, last + 1)
                cur = cur if cur and cur["article_number"] == first else None
                continue
            if col == "en" and ln["bold"]:
                # The Arabic "Second Part" heading is missing in the source; the English
                # one ("SECOND PART / REAL RIGHTS") is the only marker, so map it by hand.
                if re.match(r"^SECOND\s+PART$", text, re.I):
                    hier.update(book="الحقوق العينية", volume="", chapter="", section="", topic="")
                continue
            if col == "ar" and any(abs(y - hy) <= 4 for hy in heading_ys):
                if pending:  # first heading line after a level marker is its title
                    if NUMBERED.match(text) or ORDINAL.match(text):
                        pending = None
                    else:
                        hier[pending] = text
                        pending = None
                        continue
                m_topic = NUMBERED.match(text)
                if m_topic:
                    hier["topic"] = m_topic.group(1).strip()
                    continue
                for name, rx in AR_LEVELS:
                    if rx.match(text):
                        for lower in LEVEL_NAMES[LEVEL_NAMES.index(name) :]:
                            hier[lower] = ""
                        pending = name
                        break
                continue
            if col == "ar" and HDR_AR_JUNK.match(text) and any(abs(y - by) < 5 for by in band):
                continue  # Arabic header fragments next to an accepted English header
            if cur is not None:
                cur[col].append(text)
    return arts


def fix_ar(text: str) -> str:
    text = text.translate(TO_ASCII)
    text = re.sub(r"\d+", lambda m: m.group()[::-1], text)  # digit runs come reversed
    text = re.sub(r"[\(\)]\s*(\d+)\s*[\(\)]", r"(\1) ", text)  # paragraph markers
    text = re.sub(r"[\(\)]\s*([أ-ي])\s*[\(\)]", r"(\1) ", text)
    return re.sub(r"\s+", " ", text).strip()


def recover_gaps(arts: list[dict]) -> None:
    """Split gap articles (no English header) out of the previous article's Arabic."""
    for i, art in enumerate(arts):
        if not art["gap"] or i == 0:
            continue
        prev = arts[i - 1]
        digits = str.maketrans("0123456789", AR_DIGITS)
        tok = str(art["article_number"])[::-1].translate(digits)
        joined = " ".join(prev["ar"])
        hits = list(re.finditer(rf"(?<![٠-٩]){tok}(?![٠-٩])", joined))
        if hits:
            cut = hits[-1]
            head = re.sub(r"\s*مادة\s*[\(\)\s]*$", "", joined[: cut.start()])
            art["ar"], prev["ar"] = [joined[cut.end() :]], [head]
            art["en"] = []  # English text is merged into the previous article in the source
            art["en_missing"] = True


def apply_source_fixes(arts: list[dict]) -> None:
    """Known layout anomalies in laws.pdf (checked by hand against the page images)."""
    by = {a["article_number"]: a for a in arts}
    a53 = by.get(53)
    if a53:  # Arabic repeal notice for 54-80 starts at the bottom of article 53's row
        pattern = r"\s*(?:ألغيت\s+)?المواد\s+من\s+[٠-٩]+\s+إلى\s+[٠-٩]+\s*$"
        a53["ar"] = [re.sub(pattern, "", " ".join(a53["ar"]))]
    a1021, a1022 = by.get(1021), by.get(1022)
    if a1021 and a1022 and not any(a1022["ar"]):
        joined = " ".join(a1021["ar"])
        m = re.search(r"[\(\)]\s*٢\s*[\(\)]", joined)  # raw digits are still Arabic-Indic
        if m and m.start() > 0:
            a1021["ar"], a1022["ar"] = [joined[: m.start()]], [joined[m.start() :]]
            a1022["ar_incomplete"] = True  # first Arabic paragraph is absent in the source


def finalize(art: dict) -> dict:
    ar = fix_ar(" ".join(art["ar"]))
    en = re.sub(r"\s+", " ", " ".join(art["en"])).strip()
    n = art["article_number"]
    if art["repealed"]:
        ar, en = "ملغاة", f"Article {n} has been repealed."
    return {
        "article_id": str(n),  # kept so notebooks/00-baseline.ipynb still runs unchanged
        "article_number": n,
        "book": art["book"],
        "chapter": art["chapter"],
        "section": art["section"],
        "topic": art["topic"],
        "text_ar": ar,
        "text_en": en,
        "is_repealed": art["repealed"],
        "source_page": art["page"],
        "citation": f"Egyptian Civil Code, Article {n}",
        "volume": art["volume"],
        "en_missing": bool(art.get("en_missing")),
        "ar_incomplete": bool(art.get("ar_incomplete")),
    }


def build_corpus(pdf_path: str | Path, out_path: str | Path) -> int:
    arts = parse(pdf_path)
    recover_gaps(arts)
    apply_source_fixes(arts)
    records = [finalize(a) for a in arts]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("wrote %d articles to %s", len(records), out)
    return len(records)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the Civil Code corpus JSON from the PDF.")
    ap.add_argument("pdf", nargs="?", default=str(settings.raw_pdf))
    ap.add_argument("out", nargs="?", default=str(settings.corpus_json))
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s"
    )  # JSON logs: step 03
    build_corpus(args.pdf, args.out)


if __name__ == "__main__":
    main()
