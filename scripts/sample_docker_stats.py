"""Sample a container's CPU and memory while a load test runs.

    uv run python scripts/sample_docker_stats.py --container docker-api-1 --seconds 360 \
        --out reports/loadtest/baseline-fastapi-stats.csv

Start it just before Locust. CPU is docker's number: 100 % is one core, so 200 % means two cores
busy. It is the evidence for the bottleneck claim: which resource is saturated while latency climbs.
"""

import argparse
import csv
import statistics
import subprocess
import time
from pathlib import Path

UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}


def parse_mem(text: str) -> float:
    """'512.3MiB / 2GiB' -> MiB used."""
    used = text.split("/")[0].strip()
    for unit in ("GiB", "MiB", "KiB", "B"):
        if used.endswith(unit):
            return float(used[: -len(unit)]) * UNITS[unit] / 1024**2
    raise ValueError(f"cannot read memory from {text!r}")


def sample(container: str) -> tuple[float, float]:
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}};{{.MemUsage}}", container],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    cpu, mem = out.split(";")
    return float(cpu.rstrip("%")), parse_mem(mem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--seconds", type=int, default=360)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    end = time.time() + args.seconds
    while time.time() < end:
        cpu, mem = sample(args.container)  # takes ~1-2 s itself
        rows.append((round(time.time()), cpu, round(mem, 1)))
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unix_time", "cpu_percent", "mem_mib"])
        w.writerows(rows)
    cpus, mems = [r[1] for r in rows], [r[2] for r in rows]
    print(
        f"{len(rows)} samples | CPU% mean {statistics.mean(cpus):.0f}  max {max(cpus):.0f} "
        f"| memory MiB mean {statistics.mean(mems):.0f}  max {max(mems):.0f}"
    )


if __name__ == "__main__":
    main()
