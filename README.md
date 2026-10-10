# legal-rag

Bilingual (Arabic / English) question answering over the Egyptian Civil Code. A question is
embedded with `multilingual-e5-small` (served through ONNX Runtime), the closest articles are
retrieved from a Chroma index, and Groq writes an answer that cites the articles it used.

## Try it in 3 commands

You need Docker and a [Groq API key](https://console.groq.com/keys). The image does not contain a
key: you pass yours at run time.

```bash
docker run -d --name legal-rag -p 8000:8000 -e GROQ_API_KEY=<your key> doaamostafa679/legal-rag:0.1.0
curl localhost:8000/health        # wait about 30 s on first start while the model loads
curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"ما هو سن الرشد؟"}'
```

Example answer:

```json
{
  "answer": "بناءً على المادة 44، فقرة (2)، فإن سن الرشد هي إحدى وعشرون سنة ميلادية كاملة.",
  "sources": ["Egyptian Civil Code, Article 44", "Egyptian Civil Code, Article 46", "Egyptian Civil Code, Article 621"],
  "correlation_id": "5a63af55a1c748d799d6aae10e3af941",
  "latency_ms": 853.7
}
```

`k` (1-10, default 3) sets how many articles are retrieved: `{"question": "...", "k": 5}`.

## API

| Endpoint | What it does |
|---|---|
| `POST /ask` | Retrieve articles and generate a cited answer |
| `GET /health` | `200` with `status: healthy` once the model and index are loaded, `503` before |
| `GET /metadata` | Service version, embedding model, embedder class, documents indexed, Groq model, SHA-256 of the corpus |

Every response carries an `x-request-id` header. The same id appears as `correlation_id` in every
JSON log line of that request. Interactive docs are at `/docs`.

## Run from source

```bash
uv sync                                   # runtime + dev dependencies
echo 'GROQ_API_KEY=<your key>' > .env     # no quotes around the key
uv run python -m legal_rag.rag ingest     # build the Chroma index from data/processed/civil_code.json
uv run uvicorn legal_rag.api.main:app --port 8000
```

The ONNX encoder is built once with `uv run python -m legal_rag.export` (needs the `export` and
`offline` dependency groups, which `uv sync` installs). `uv run python -m legal_rag.rag query "..."`
retrieves articles without calling Groq.

With Docker Compose:

```bash
docker compose --env-file .env -f docker/docker-compose.yml up --build
```

Settings come from environment variables (prefix `LEGAL_RAG_`) or `.env`; `GROQ_API_KEY` is read
as is. `LEGAL_RAG_ROOT` moves `data/`, `chroma_db/` and `models/` (the image sets it to `/app`).

## Development

```bash
uv run pytest                              # 77 unit tests, coverage gate 70%
uv run pytest -m integration --no-cov      # real model: hit@3 vs notebook, ONNX parity (~6 min)
uv run ruff check . && uv run ruff format --check .
uv run pre-commit install
```

## Results (Module 1)

| | |
|---|---|
| Retrieval, 10 questions (5 AR + 5 EN) | hit@3 = 0.9 |
| Encoder latency, CPU, batch 1 | PyTorch 60.6 ms mean, ONNX Runtime 55.3 ms mean |
| `.venv` without PyTorch | 424 MB (1.7 GB with it) |
| Docker image | 489 MB compressed, runs as non-root, has a healthcheck |

Details, the serialization comparison and the limits are in `reports/module-1.md`.

## Layout

```
src/legal_rag/
  api/            FastAPI app and request/response models
  rag.py          ingest + retrieve (Chroma), CLI
  onnx_embed.py   ONNX encoder with mean pooling and L2 norm
  generate.py     Groq answer generation
  parse_civil_code.py  PDF -> data/processed/civil_code.json
  export.py, benchmark.py  PyTorch -> ONNX export and latency benchmark
  config.py, logging_conf.py, timing.py
tests/            unit tests (mocked embedder and Groq) + integration tests
docker/           Dockerfile, single-stage comparison file, compose file
reports/          module reports
```

## Limits

Answers come only from the Civil Code (1149 articles, 1093 indexed; 56 repealed articles are
excluded). The evaluation set has 10 questions, so 0.9 is a signal, not a benchmark. Answers are
generated in Arabic whatever the question language. This is a retrieval demo, not legal advice.
