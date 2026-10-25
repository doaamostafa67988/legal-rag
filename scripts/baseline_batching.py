"""CAT-1 problem 1: does batching help this encoder?

    uv run python scripts/baseline_batching.py --out reports/baseline-batching.md

/ask embeds one question per call. This runs the same encoder over the same questions with
different batch sizes, in two ways: questions batched as they arrive, and questions grouped by
length first (a batch is padded to its longest question, so mixed lengths waste work). Same model,
same inputs: the difference is what batching buys, or costs, on this CPU.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = (1, 2, 4, 8, 16, 32, 100)
REPEATS = 5


def questions(n: int) -> list[str]:
    dev = json.loads((ROOT / "data/eval/questions.json").read_text(encoding="utf-8"))
    corpus = json.loads((ROOT / "data/processed/civil_code.json").read_text(encoding="utf-8"))
    articles = corpus if isinstance(corpus, list) else corpus["articles"]
    pool = [q["question"] for q in (dev if isinstance(dev, list) else dev["questions"])]
    pool += ["ما حكم: " + " ".join(a["text_ar"].split()[:15]) for a in articles if a["text_ar"]]
    return pool[:n]


def total_seconds(embed: Callable[[list[str]], object], qs: Sequence[str], size: int) -> float:
    start = time.perf_counter()
    for i in range(0, len(qs), size):
        embed(list(qs[i : i + size]))
    return time.perf_counter() - start


def sweep(
    embed: Callable[[list[str]], object],
    qs: Sequence[str],
    sizes: Sequence[int] = SIZES,
    repeats: int = REPEATS,
) -> list[dict]:
    """Median seconds to embed all of `qs` for each batch size, as-arrived and length-grouped."""
    grouped = sorted(qs, key=len)
    rows = []
    for size in sizes:
        arrived = statistics.median(total_seconds(embed, qs, size) for _ in range(repeats))
        by_len = statistics.median(total_seconds(embed, grouped, size) for _ in range(repeats))
        rows.append({"size": size, "arrived": arrived, "grouped": by_len})
    return rows


def table(rows: list[dict], n: int) -> str:
    base = rows[0]["arrived"]
    lines = [
        f"{n} questions, ONNX Runtime CPU, median of {REPEATS} runs; "
        "ms per question (lower is better)",
        "",
        "| Batch size | as they arrive | grouped by length | speed-up vs batch 1 |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['arrived'] / n * 1000:.1f} | {r['grouped'] / n * 1000:.1f} "
            f"| {base / r['arrived']:.2f}x |"
        )
    best = min(rows, key=lambda r: r["arrived"])
    lines += ["", f"Fastest batch size as the questions arrive: {best['size']}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    from legal_rag.onnx_embed import OnnxEmbedder

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    qs = questions(args.n)
    enc = OnnxEmbedder()
    enc.embed_queries(qs[:10])  # warm-up: the first calls pay one-off initialisation costs
    text = table(sweep(enc.embed_queries, qs), len(qs))
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
