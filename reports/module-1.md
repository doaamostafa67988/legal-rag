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
