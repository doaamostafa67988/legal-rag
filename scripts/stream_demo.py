"""Show that /ask_stream sends the answer while it is being written.

    uv run python scripts/stream_demo.py                      # BentoML on :3000
    uv run python scripts/stream_demo.py --question "ما هو سن الرشد؟" --k 3

Prints each piece of the answer with the time it arrived, then the time to the first piece and to
the end. With /ask the same answer arrives in one piece, after the end.
"""

import argparse
import json
import time
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3000/ask_stream")
    ap.add_argument("--question", default="ما هو سن الرشد؟")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    body = json.dumps({"question": args.question, "k": args.k}).encode()
    request = urllib.request.Request(
        args.url, data=body, headers={"content-type": "application/json"}
    )
    start = time.perf_counter()
    first = None
    pieces = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        for piece in iter(lambda: response.read1(64), b""):
            now = time.perf_counter() - start
            first = first if first is not None else now
            pieces += 1
            print(f"{now:6.2f}s  {piece.decode('utf-8', 'replace')!r}")
    end = time.perf_counter() - start
    print(f"\nfirst piece after {first:.2f}s, finished after {end:.2f}s, {pieces} pieces")


if __name__ == "__main__":
    main()
