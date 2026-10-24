"""Ask the running API every English dev question and score the cited articles.

    uv run python scripts/probe_serving.py            # prints one line per question + a total
    uv run python scripts/probe_serving.py --out reports/step3-v1.txt

It reports which retriever answered (GET /model) and hit@3 on the 15 English dev questions,
the questions an Arabic-only index handles worst. Run it before and after moving the alias.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

QUESTIONS = Path("data/eval/questions.json")


def call(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def cites(sources: list[str], article: int) -> bool:
    return any(re.search(rf"\bArt(?:icle|\.)?\s*{article}\b", s) for s in sources)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    model = call(f"{args.url}/model")
    lines = [
        f"retriever: {model['source']} v{model.get('version')} "
        f"variants={model['variants']} fusion={model['fusion']}"
    ]
    questions = [q for q in json.loads(QUESTIONS.read_text()) if q["lang"] == args.lang]
    hits = 0
    for q in questions:
        sources = call(f"{args.url}/ask", {"question": q["question"], "k": 3})["sources"]
        ok = any(cites(sources, g) for g in q["gold_articles"])
        hits += ok
        verdict = "HIT " if ok else "MISS"
        lines.append(f"{verdict} {q['id']} gold={q['gold_articles']} got={sources}")
    lines.append(f"hit@3 ({args.lang}) = {hits}/{len(questions)} = {hits / len(questions):.3f}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
