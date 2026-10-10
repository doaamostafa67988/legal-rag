"""Latency benchmark: PyTorch (eager) vs ONNX Runtime on the same 500 articles, batch size 1."""

import statistics
import sys
import time

from legal_rag.data import load_corpus
from legal_rag.onnx_embed import OnnxEmbedder
from legal_rag.rag import E5Embedder

N = 500
WARMUP = 10


def _time_ms(embedder, texts: list[str]) -> list[float]:
    for t in texts[:WARMUP]:  # warm-up: first calls pay one-off initialisation costs
        embedder.embed_passages([t])
    times = []
    for t in texts:
        start = time.perf_counter()
        embedder.embed_passages([t])
        times.append((time.perf_counter() - start) * 1000)
    return times


def _row(name: str, times: list[float]) -> str:
    p95 = statistics.quantiles(times, n=100)[94]
    return f"| {name} | {statistics.mean(times):.1f} | {p95:.1f} |"


def main() -> None:
    articles = [a for a in load_corpus() if not a.is_repealed and a.text_ar.strip()]
    texts = [a.text_ar for a in articles[:N]]
    lines = [
        f"Latency per article, batch size 1, n={len(texts)}",
        "",
        "| Runtime | mean (ms) | p95 (ms) |",
        "|---|---|---|",
        _row("PyTorch (eager)", _time_ms(E5Embedder(), texts)),
        _row("ONNX Runtime (CPU)", _time_ms(OnnxEmbedder(), texts)),
    ]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
