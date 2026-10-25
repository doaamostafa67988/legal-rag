"""BentoML service for the Civil Code RAG (Module 3, Step 4).

    uv run bentoml serve serving.service:LegalRAGService          # http://localhost:3000

Two services, two worker processes:

  Encoder          embeds questions with the ONNX model. Adaptive micro-batching: questions that
                   arrive within `max_latency_ms` of each other go through the model in ONE call
                   (up to `max_batch_size`). It runs in its own process, so its CPU work is out
                   from under the API process's GIL (BentoML 1.4's replacement for the Runner).
  LegalRAGService  the API. `ask` and `ask_stream` are `async`: the wait for the encoder and for
                   the LLM is an `await`, so one process serves many requests at once. It ranks
                   with the retriever (the registry's `@production` version, or the baked index).

Calls: POST /ask {"question": "...", "k": 3}  ·  POST /ask_stream (tokens as they are written)
       POST /metadata, POST /model  ·  GET /livez, /readyz, /metrics (built in)

The trade-off in `max_batch_size` / `max_latency_ms`: a bigger batch makes the encoder more
efficient per question (throughput), but the dispatcher may hold a request to batch it with others.
`max_latency_ms` is the latency the encoder step is allowed to take, queueing included: when
BentoML predicts a request cannot finish within it, it answers 503 ("process is overloaded") instead
of letting the queue grow. Too small a value (10 ms was tried) sheds load even when the service is
healthy; too large a value hides overload behind slow answers. Tune both with BENTO_MAX_BATCH_SIZE
and BENTO_MAX_LATENCY_MS and measure each setting with the same Locust ramp. BentoML needs
max_batch_size > 1; BENTO_BATCHING=off turns batching off for the control run.
"""

import asyncio
import hashlib
import logging
import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

# a built Bento keeps the repo's `src/` next to `serving/`; a plain checkout has it too
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import aiohttp  # noqa: E402  (BentoML's own HTTP client between its services)
import bentoml  # noqa: E402
import numpy as np  # noqa: E402

from legal_rag.api.schemas import AskRequest  # noqa: E402
from legal_rag.config import settings  # noqa: E402
from legal_rag.generate import StubChat, answer_async, stream_answer  # noqa: E402
from legal_rag.logging_conf import configure_logging  # noqa: E402
from legal_rag.rag import LegalRAG, sources_of  # noqa: E402
from legal_rag.registry_rag import load_from_registry  # noqa: E402

log = logging.getLogger("legal_rag.serving")

MAX_BATCH_SIZE = int(os.environ.get("BENTO_MAX_BATCH_SIZE", "16"))
MAX_LATENCY_MS = int(os.environ.get("BENTO_MAX_LATENCY_MS", "1000"))
# BENTO_BATCHING=off is the control for the batching experiment: same two processes and same async
# API, but every question goes through the encoder alone (BentoML refuses max_batch_size=1).
BATCHING = os.environ.get("BENTO_BATCHING", "on").lower() != "off"
# At most this many questions wait on the Encoder at once; the rest wait here, in the API's event
# loop, which costs nothing. Without a cap, 150 users open 150 connections to the Encoder and
# BentoML's inter-service client starts failing (see `_embed`).
ENCODER_CONCURRENCY = int(os.environ.get("BENTO_ENCODER_CONCURRENCY", "16"))
ENCODER_RETRIES = int(os.environ.get("BENTO_ENCODER_RETRIES", "2"))
_EMBED_API = (
    bentoml.api(batchable=True, max_batch_size=MAX_BATCH_SIZE, max_latency_ms=MAX_LATENCY_MS)
    if BATCHING
    else bentoml.api()
)


@bentoml.service(
    name="legal_rag_encoder",
    resources={"cpu": "1"},
    traffic={"timeout": 30},
)
class Encoder:
    def __init__(self) -> None:
        from legal_rag.onnx_embed import OnnxEmbedder

        self.model = OnnxEmbedder()  # loaded once per worker, never per request

    @_EMBED_API
    def embed(self, questions: list[str]) -> np.ndarray:
        """One row per question; BentoML merges concurrent calls and splits the result back."""
        return np.asarray(self.model.embed_queries(questions), dtype=np.float32)


class RemoteEmbedder:
    """What the retriever sees as its `embedder`: the vectors come from the Encoder service."""

    def __init__(self, encoder) -> None:
        self._encoder = encoder

    def embed_query(self, text: str) -> list[float]:
        return self._encoder.embed([text])[0].tolist()


def _sha256(path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _chat():
    if settings.chat_backend == "stub":
        log.warning("serving with the STUB language model: load testing only")
        return StubChat()
    from legal_rag.generate import GroqChat

    return GroqChat()


@bentoml.service(
    name="legal_rag",
    resources={"cpu": "1"},
    traffic={"timeout": 60},
)
class LegalRAGService:
    encoder = bentoml.depends(Encoder)

    def __init__(self) -> None:
        configure_logging(settings.log_level)
        self._encoder_slots = asyncio.Semaphore(ENCODER_CONCURRENCY)
        embedder = RemoteEmbedder(self.encoder)
        if settings.registry_model_uri:  # the registry decides what is served
            self.rag = load_from_registry(settings.registry_model_uri, embedder)
        else:
            self.rag = LegalRAG(embedder=embedder)
        self.chat = _chat()
        self.corpus_sha256 = _sha256(settings.corpus_json)
        log.info("service ready", extra={"documents_indexed": self.rag.documents_indexed})

    async def _embed(self, question: str) -> np.ndarray:
        """Ask the Encoder for the vector.

        Two protections, both added after a 150-user test failed 27 % of requests with
        `ServerDisconnectedError` / `Connector is closed` raised by BentoML's inter-service client
        while the Encoder itself logged only 200s: (1) a cap on concurrent calls to the Encoder,
        so a burst waits in the API instead of opening one connection per user; (2) a retry when
        the pooled connection is dropped anyway. Embedding is a pure function of the question, so
        repeating the call is safe.
        """
        async with self._encoder_slots:
            for attempt in range(ENCODER_RETRIES + 1):
                try:
                    return (await self.encoder.to_async.embed([question]))[0]
                except aiohttp.ClientConnectionError:
                    if attempt == ENCODER_RETRIES:
                        raise
                    log.warning("encoder connection dropped, retrying")
                    await asyncio.sleep(0.02 * (attempt + 1))
        raise AssertionError("unreachable")

    async def _retrieve(self, body: AskRequest):
        vector = await self._embed(body.question)
        return self.rag.search(vector, body.k, question_chars=len(body.question))

    @staticmethod
    def _validated(question: str, k: int) -> AskRequest:
        from pydantic import ValidationError

        try:
            return AskRequest(question=question, k=k)
        except ValidationError as exc:
            raise bentoml.exceptions.InvalidArgument(str(exc)) from exc

    @bentoml.api
    async def ask(self, question: str, k: int = 3) -> dict:
        body = self._validated(question, k)
        hits = await self._retrieve(body)
        text = await answer_async(body.question, hits, self.chat)
        return {"answer": text, "sources": sources_of(hits)}

    @bentoml.api
    async def ask_stream(self, question: str, k: int = 3) -> AsyncGenerator[str, None]:
        """The answer as it is written; the sources come last, on their own line."""
        body = self._validated(question, k)
        hits = await self._retrieve(body)
        async for token in stream_answer(body.question, hits, self.chat):
            yield token
        yield "\n\nالمصادر: " + "، ".join(sources_of(hits))

    @bentoml.api
    def metadata(self) -> dict:
        return {
            "embedding_model": settings.embedding_model,
            "embedder": type(self.rag.embedder).__name__,
            "documents_indexed": self.rag.documents_indexed,
            "llm": "stub" if settings.chat_backend == "stub" else settings.groq_model,
            "corpus_sha256": self.corpus_sha256,
            "batching": BATCHING,
            "encoder_concurrency": ENCODER_CONCURRENCY,
            "max_batch_size": MAX_BATCH_SIZE,
            "max_latency_ms": MAX_LATENCY_MS,
        }

    @bentoml.api
    def model(self) -> dict:
        """Which retriever answers: a registry version, or the index baked into the image."""
        return getattr(self.rag, "info", None) or {
            "source": "chroma",
            "embedding_model": settings.embedding_model,
            "variants": ["ar"],
            "fusion": "single",
        }
