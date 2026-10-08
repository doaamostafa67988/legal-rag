# Module 1 — Baseline

## Setup
- Embedding: intfloat/multilingual-e5-small, Chroma (cosine), 1 chunk per article
- Corpus: 1149 articles of the Egyptian Civil Code (data/processed/civil_code.json)
- LLM (Groq): qwen/qwen3.8-27b (output capped with max_completion_tokens)

## Retrieval evaluation
- 10 questions (5 AR + 5 EN, same 5 articles)
- hit@3 = 0.9 (9/10)
- English queries: 5/5, Arabic queries: 4/5
- The only miss: the Arabic question for Art. 163 returned 168, 166, 521

## Answer-quality spot checks (Groq, k=3)
- AR "age of majority": correct, cites Art. 44.
- EN "age of majority": answered in Arabic (system prompt forces Arabic) — language policy still to decide.
- AR Art. 163 question: retrieval missed 163; the LLM refused to answer from the
  irrelevant articles instead of hallucinating. This is a retrieval failure, not a
  generation failure, so a reranker is the candidate fix (later module).
- Out-of-domain ("capital of France"): refused correctly, but the reply is verbose.

## Limits
- Only 10 evaluation questions: 0.9 vs 1.0 is a single question, so treat it as a signal only.
- Answers come only from the corpus; other laws are not covered.

## Step 2 - Package refactor

- Notebook logic moved into `src/legal_rag/` (ingest, retrieve and generate behind small interfaces).
- hit@3 after the refactor: `test_baseline_hit_at_3_matches_the_notebook` passes, so the package
  gives the same 0.9 (9/10) as the notebook.
- `@timed` decorator (`timing.py`) on `LegalRAG.retrieve`; it logs the duration at DEBUG so the
  INFO latency line is not duplicated.

## Step 3 - Logging

- JSON logs with a correlation id set per request (`X-Request-ID`, or generated).
- Check: one `/ask` call returned `x-request-id: 100c9503...`; the same id appears in the three log
  lines `question served`, `answer generated` and `request`.
- Question and answer text are not logged at INFO (only their length).

## Step 4 - Serialization (embedding model: PyTorch vs ONNX)

- Exported the multilingual-e5-small encoder with `legal_rag.export` (dynamic axes on batch and
  sequence); mean pooling and L2 normalisation stay outside the graph (`onnx_embed.py`).
- Parity test (`tests/test_serialization.py`): 500 articles through PyTorch and ONNX,
  `np.allclose(atol=1e-4)` passes, min cosine similarity > 0.9999. A second test checks that a
  batch of 7 gives the same vectors as 7 single calls (the dynamic batch axis works).
- Benchmark (`python -m legal_rag.benchmark`): CPU only, WSL2, batch size 1, n=500, 10 warm-up calls.

| Runtime | mean (ms) | p95 (ms) |
|---|---|---|
| PyTorch (eager) | 60.6 | 109.7 |
| ONNX Runtime (CPU) | 55.3 | 120.8 |

ONNX is about 9% faster on the mean but slower at p95, so on this machine the two are roughly
equal. Speed is not the reason to choose ONNX here.

### Format comparison

| Format | Human-readable | Cross-language | Schema-enforced | Safe to load from an untrusted source |
|---|---|---|---|---|
| JSON | Yes | Yes | No (only with a separate JSON Schema) | Yes (data only) |
| Protobuf | No (binary) | Yes | Yes (`.proto` schema) | Yes (data only) |
| Pickle | No | No (Python only) | No | **No: executes arbitrary code on load** |
| ONNX | No (binary protobuf, viewable in Netron) | Yes | Yes (typed graph, `onnx.checker`) | Mostly: data-only graph, no code execution, but custom ops and external data files still need trust |

Pickle executes arbitrary code on load. Never load a `.pkl` (or a PyTorch `.pt`/`.bin`
checkpoint, which uses pickle) that you did not produce yourself.

**Serving format: ONNX**, because it is cross-language, schema-checked and loads without running
code, and it lets the serving path run on onnxruntime without PyTorch (to confirm when the Docker
image is built); latency is about the same as eager PyTorch.

## Step 5 - API

- `POST /ask`, `GET /health`, `GET /metadata`. The embedder and the index load once in the lifespan,
  never per request.
- Invalid input returns a 422 with readable per-field errors; unexpected errors return a clean 500
  with the correlation id and no traceback.
- `/metadata` reports the service version, embedding model, embedder class (`OnnxEmbedder` when
  served), documents indexed (1093 = 1149 articles minus 56 repealed), Groq model and the SHA-256 of
  the corpus file.
- Latency in two sample requests: retrieval 16-31 ms, Groq generation 450-600 ms, total about
  470-640 ms. The LLM call dominates, so the embedder runtime matters little for end-to-end time.

## Step 6 - Tests

- 74 unit tests run by default. 3 integration tests (hit@3 against the notebook, ONNX parity,
  dynamic batch axis) run with `pytest -m integration`; they take about 6 minutes because they load
  the real model.
- Coverage is 75% against a 70% gate. The lowest module is `parse_civil_code.py` at 34%: its PDF
  paths need the source PDF, while its pure functions are unit-tested.
- Mocks: a fake embedder, a fake chat client and a fake `groq` module, so tests need no network and
  no API key.
- Break check: changed `sources_of` to return `str(h.article_number)`; two tests failed
  (`test_sources_are_article_citations_not_chunk_ids` in `test_rag.py` and
  `test_ask_returns_answer_and_article_citations` in `test_api.py`), then restored with
  `git restore`; all 74 tests green again.
- Tooling: `ruff format` is used instead of `black`. Pre-commit runs ruff, end-of-file,
  trailing-whitespace and line-ending hooks.

## Dependency split

- Base dependencies are what the API needs to serve; `torch`, `sentence-transformers` and `pymupdf`
  moved to the `offline` dependency group (dev installs them, the image does not).
- `.venv` size: 1.7 GB with everything, 424 MB with `uv sync --no-dev`.
- Check: with `--no-dev` the service starts, `/health` returns 200 and `/ask` answers, so serving
  does not need PyTorch.
- `LEGAL_RAG_ROOT` sets where `data/`, `chroma_db/` and `models/` live; the default is the
  checkout, so an installed package (Docker) no longer resolves paths inside `site-packages`.
