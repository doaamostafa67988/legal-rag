"""Load test for the /ask service.

    uv run locust -f loadtest/locustfile.py --host http://localhost:8000          # web UI on :8089
    uv run locust -f loadtest/locustfile.py --host http://localhost:8000 --headless \
        -u 50 -r 2 -t 6m --csv reports/loadtest/baseline-fastapi

Mix, from the handbook adapted to this API (there is no batch endpoint): 80 % ordinary questions,
15 % questions that ask for ten articles (the heavy case: a long prompt), 5 % /metadata and /model.
Questions are drawn at random from a pool of dev questions and questions cut from random articles,
in Arabic and English, so no request is repeated often enough for a cache to flatter the result.

Against the BentoML service set LOADTEST_TARGET=bento (its /metadata is a POST).

Serve with the stub LLM while testing (docker/docker-compose.loadtest.yml), or the numbers
measure the LLM provider's rate limits instead of this service.
"""

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
SEED_POOL = 300
# BentoML exposes every endpoint as POST; the FastAPI service has GET /metadata.
METADATA_POST = os.environ.get("LOADTEST_TARGET") == "bento"


def build_pool(rng: random.Random) -> list[str]:
    """Dev questions plus questions made from the first 6-30 words of random articles."""
    dev = json.loads((ROOT / "data/eval/questions.json").read_text(encoding="utf-8"))
    pool = [q["question"] for q in (dev if isinstance(dev, list) else dev["questions"])]
    corpus = json.loads((ROOT / "data/processed/civil_code.json").read_text(encoding="utf-8"))
    articles = corpus if isinstance(corpus, list) else corpus["articles"]
    texts = [a["text_ar"] for a in articles if a.get("text_ar", "").strip()]
    for text in rng.sample(texts, min(SEED_POOL, len(texts))):
        words = text.split()
        pool.append("ما حكم: " + " ".join(words[: rng.randint(6, 30)]))
    return pool


POOL = build_pool(random.Random(2026))


class AskUser(HttpUser):
    wait_time = between(1, 3)

    def _ask(self, k: int, name: str) -> None:
        with self.client.post(
            "/ask", json={"question": random.choice(POOL), "k": k}, name=name, catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            body = resp.json()
            if not body.get("answer", "").strip() or not body.get("sources"):
                resp.failure("empty answer or no sources")

    @task(80)
    def ask(self) -> None:
        self._ask(3, "/ask k=3")

    @task(15)
    def ask_ten_articles(self) -> None:
        self._ask(10, "/ask k=10")

    @task(5)
    def metadata(self) -> None:
        if METADATA_POST:
            self.client.post("/metadata", json={}, name="/metadata")
        else:
            self.client.get("/metadata", name="/metadata")
