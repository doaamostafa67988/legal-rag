"""Print the numbers for the report from a Locust stats CSV, as a markdown table.

    uv run python scripts/summarize_loadtest.py reports/loadtest/baseline-fastapi_stats.csv

Locust writes <prefix>_stats.csv when you pass --csv <prefix>. Times are in milliseconds.
"""

import csv
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: summarize_loadtest.py <locust _stats.csv>")
    with Path(sys.argv[1]).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print("| Endpoint | requests | failures | p50 (ms) | p95 (ms) | p99 (ms) | RPS |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        n, failed = int(r["Request Count"]), int(r["Failure Count"])
        print(
            f"| {r['Name']} | {n} | {failed} ({failed / n:.1%}) | {r['50%']} | {r['95%']} "
            f"| {r['99%']} | {float(r['Requests/s']):.2f} |"
        )


if __name__ == "__main__":
    main()
