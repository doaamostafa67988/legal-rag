"""Parse the bilingual Egyptian Civil Code PDF into one JSON record per article.

Facts about laws.pdf (verified):
- Two columns: English on the left, Arabic on the right, one table row per article.
- The English header "Article N" is the reliable anchor (Arabic digit runs are
  reversed by the PDF export, e.g. 13 -> 31, 1956 -> 6591).
- Word stores the lam-alef ligature reversed ("la" -> "al"); the swapped alef has a
  zero-width bbox, which distinguishes it from the normal definite article.
- Some English headers are missing in the source (e.g. 452, 1022): recovered from
  the Arabic header digits that leak into the previous article.
- Repeal notices ("Articles 54-80 ... repealed") become stub records.
"""

import json
import re
import sys
from pathlib import Path

import pymupdf

ALEFS = set("اأإآ")
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
TO_ASCII = str.maketrans(AR_DIGITS, "0123456789")
HDR_EN = re.compile(r"^Article\s*(\d+)\b\s*(.*)$")
REPEAL = re.compile(r"^\*?\s*Articles?\s+(\d+)\s*(?:-|to)\s*(\d+)\b.*repealed", re.I)
HDR_AR_JUNK = re.compile(r"^[\(\)\s\d٠-٩]*(?:مادة)?[\(\)\s\d٠-٩]*$")
LEVELS = [
    ("part", re.compile(r"^(FIRST|SECOND)\s+PART$", re.I)),
    ("book", re.compile(r"^BOOK\s+[IVX]+$", re.I)),
    ("chapter", re.compile(r"^CHAPTER\s+[IVX]+$", re.I)),
    ("section", re.compile(r"^SECTION\s+[IVX]+$", re.I)),
]


def fix_ligatures(chars):
    """Put swapped lam-alef ligatures back in order (swapped alef has zero width)."""
    out = [c["c"] for c in chars]
    for i in range(len(chars) - 1):
        a, n = chars[i], chars[i + 1]
        if a["c"] in ALEFS and n["c"] == "ل" and (a["bbox"][2] - a["bbox"][0]) < 0.3:
            out[i], out[i + 1] = "ل", a["c"]
    return "".join(out)


def page_lines(page):
    """Yield dicts: col, y, text, bold (majority of chars in a bold font)."""
    mid = page.rect.width / 2
    for b in page.get_text("rawdict")["blocks"]:
        for line in b.get("lines", []):
            chars = [c for s in line["spans"] for c in s["chars"]]
            x0, y0, x1, _ = line["bbox"]
            col = "en" if (x0 + x1) / 2 < mid else "ar"
            if col == "ar":  # stream order can misplace punctuation; use visual RTL order
                chars.sort(key=lambda c: -(c["bbox"][0] + c["bbox"][2]) / 2)
            text = fix_ligatures(chars).strip()
            if not text:
                continue
            tot = sum(1 for c in chars if c["c"].strip())
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
                "text": text,
                "bold": tot > 0 and bold / tot >= 0.5,
            }


def order_lines(lines):
    """Sort by y; fragments of one visual line (same column, |dy|<2.5) go left->right
    for English and right->left for Arabic."""
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
        run = lines[i:j]
        run.sort(key=lambda d: d["x0"] if d["col"] == "en" else -d["x0"])
        merged = run[0]
        for nxt in run[1:]:  # glue fragments split mid-word (tiny horizontal gap)
            gap = nxt["x0"] - merged["x1"] if nxt["col"] == "en" else merged["x0"] - nxt["x1"]
            sep = "" if gap < 1.5 else " "
            merged = {
                **merged,
                "text": merged["text"] + sep + nxt["text"],
                "x0": min(merged["x0"], nxt["x0"]),
                "x1": max(merged["x1"], nxt["x1"]),
                "bold": merged["bold"] and nxt["bold"],
            }
        out.append(merged)
        i = j
    return out


def new_article(n, page, hier, notice=False):
    return {
        "article_number": n,
        "page": page,
        "ar": [],
        "en": [],
        "gap": False,
        "repealed": notice,
        **{k: v for k, v in hier.items()},
    }


def parse(pdf_path):
    doc = pymupdf.open(pdf_path)
    arts, cur, expected = [], None, 1
    hier = {"part": "", "book": "", "chapter": "", "section": ""}
    pending = None  # heading level waiting for its title line
    for pno, page in enumerate(doc):
        lines = order_lines(page_lines(page))
        # pass A: accepted English headers on this page (strict sequence check)
        headers, exp = [], expected
        for ln in lines:
            if ln["col"] == "en" and REPEAL.match(ln["text"]):
                exp = max(exp, int(REPEAL.match(ln["text"]).group(2)) + 1)
                continue
            m = HDR_EN.match(ln["text"]) if ln["col"] == "en" else None
            if not m:
                continue
            n = int(m.group(1))
            if n == exp or n == exp + 1:
                headers.append((ln["y"], n, m.group(2)))
                exp = n + 1
        # pass B: assign lines
        band = [h[0] for h in headers]
        # English bold lines (not headers/notices) are headings; Arabic lines on the
        # same rows are headings too. Arabic body can be bold, so match by row, not font.
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
            if (
                col == "en"
                and hi < len(headers)
                and abs(headers[hi][0] - y) < 1.0
                and HDR_EN.match(text)
                and int(HDR_EN.match(text).group(1)) == headers[hi][1]
            ):
                _, n, rest = headers[hi]
                hi += 1
                if n == expected + 1:  # English header missing for `expected`
                    g = new_article(expected, pno + 1, hier)
                    g["gap"] = True
                    arts.append(g)
                cur = new_article(n, pno + 1, hier)
                if rest:
                    cur["en"].append(rest)
                arts.append(cur)
                expected = n + 1
                continue
            m = REPEAL.match(text) if col == "en" else None
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                for k in range(a, b + 1):
                    if cur is not None and cur["article_number"] == k:
                        cur["repealed"] = True
                    elif k >= expected:
                        s = new_article(k, pno + 1, hier, notice=True)
                        arts.append(s)
                expected = max(expected, b + 1)
                cur = cur if cur and cur["article_number"] == a else None
                continue
            if col == "en" and ln["bold"]:  # headings -> hierarchy metadata
                if pending:
                    hier[pending] = text
                    pending = None
                else:
                    for name, rx in LEVELS:
                        if rx.match(text):
                            hier[name] = text.title()
                            pending = name
                            for lower in [n for n, _ in LEVELS][
                                [n for n, _ in LEVELS].index(name) + 1 :
                            ]:
                                hier[lower] = ""
                            break
                continue
            if col == "ar" and any(abs(y - hy) <= 4 for hy in heading_ys):
                continue
            if col == "ar" and HDR_AR_JUNK.match(text) and any(abs(y - by) < 5 for by in band):
                continue  # Arabic header fragments next to an accepted English header
            if cur is not None:
                cur[col].append(text)
    return arts


def fix_ar(text):
    text = text.translate(TO_ASCII)
    text = re.sub(r"\d+", lambda m: m.group()[::-1], text)  # digit runs come reversed
    text = re.sub(r"[\(\)]\s*(\d+)\s*[\(\)]", r"(\1) ", text)  # paragraph markers
    text = re.sub(r"[\(\)]\s*([أ-ي])\s*[\(\)]", r"(\1) ", text)
    return re.sub(r"\s+", " ", text).strip()


def recover_gaps(arts):
    """Split gap articles (no English header) out of the previous article's Arabic."""
    for i, a in enumerate(arts):
        if not a["gap"] or i == 0:
            continue
        prev = arts[i - 1]
        tok = str(a["article_number"])[::-1].translate(str.maketrans("0123456789", AR_DIGITS))
        joined = " ".join(prev["ar"])
        hits = [m for m in re.finditer(rf"(?<![٠-٩]){tok}(?![٠-٩])", joined)]
        if hits:
            cut = hits[-1]
            head = re.sub(r"\s*مادة\s*[\(\)\s]*$", "", joined[: cut.start()])
            a["ar"], prev["ar"] = [joined[cut.end() :]], [head]
            a["en"] = []  # English text is merged into the previous article in the source
            a["en_missing"] = True


def apply_source_fixes(arts):
    """Known layout anomalies in laws.pdf (verified by hand against the page images)."""
    by = {a["article_number"]: a for a in arts}
    a1021, a1022 = by.get(1021), by.get(1022)
    if a1021 and a1022 and not any(a1022["ar"]):
        joined = " ".join(a1021["ar"])
        m = re.search(r"[\(\)]\s*٢\s*[\(\)]", joined)  # raw digits are still Arabic-Indic here
        cut = m.start() if m else -1
        if cut > 0:
            a1021["ar"], a1022["ar"] = [joined[:cut]], [joined[cut:]]
            a1022["ar_incomplete"] = True  # first Arabic paragraph is absent in the source


def finalize(a, doc_name="Egyptian Civil Code"):
    ar, en = fix_ar(" ".join(a["ar"])), re.sub(r"\s+", " ", " ".join(a["en"])).strip()
    n = a["article_number"]
    if a.get("repealed"):
        ar, en = "ملغاة", f"Article {n} has been repealed."
    ar = re.sub(r"\s*\(?\d+\)?$", "", ar) if re.search(r"\s\d{1,4}$", ar) and False else ar
    return {
        "article_id": str(n),
        "article_number": n,
        "page": a["page"],
        "part": a["part"],
        "book": a["book"],
        "chapter": a["chapter"],
        "section": a["section"],
        "text_ar": ar,
        "text_en": en,
        "is_repealed": bool(a.get("repealed")),
        "en_missing": bool(a.get("en_missing")),
        "ar_incomplete": bool(a.get("ar_incomplete")),
        "citation": f"{doc_name}, Art. {n}",
    }


if __name__ == "__main__":
    arts = parse(sys.argv[1])
    recover_gaps(arts)
    apply_source_fixes(arts)
    out = [finalize(a) for a in arts]
    dst = Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(out)} articles -> {dst}")
