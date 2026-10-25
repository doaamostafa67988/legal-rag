# Module 3 — Inference, serving and release strategies

Branch: `module-3-serving-release` · Tag after merge: `v0.3.0`

> Draft of the written part (Step 2) and the plan. Numbers marked *proposed* are targets to confirm;
> the vocabulary and the stack diagram should be rewritten in your own words before you submit.

## Starting point (what exists before Module 3)

- `/ask` is a plain `def` endpoint in FastAPI (run in a worker thread). One call = embed the question with the ONNX encoder (`multilingual-e5-small`, CPU), rank the registered vectors, then **one blocking call to a hosted LLM (Groq)**.
- The encoder costs 18 ms per question on CPU for the short questions of the load test (measured in `reports/baseline-batching.md`, one call per question); the Module 1 benchmark measured 55.3 ms with ONNX Runtime and 60.6 ms with PyTorch eager on other inputs, so the two numbers are not mixed. The README example response shows about 850 ms end to end, so the LLM call is most of the time (to be confirmed with traces in Module 5).
- The retriever is served from the MLflow registry (`models:/legal-rag-retriever@production`); `GET /model` says which version answers.
- Because generation is a hosted API, a load test of `/ask` can end up measuring Groq's rate limits and network, not this service. Step 3 and Step 8 need a way to separate the two (see the plan below).

## Step 2 — Inference patterns, chosen before any serving code

### Three failure modes

| Failure mode | Example for this system |
|---|---|
| Cost blowout | Putting the 10 longest articles into every prompt of a hosted LLM, which is billed per token; or keeping a GPU vLLM instance running 24 h for a service used a few hours a day |
| Latency SLA breach | `/ask` returns only when the whole 400-token answer is written: the user stares at nothing for seconds although retrieval took about 55 ms |
| Over-engineering | Kafka, Triton and Kubernetes for a corpus of 1,149 articles and a retriever that runs on a CPU |

### Decision table

| Scenario | Pattern | The SLA that drives it |
|---|---|---|
| (a) A lawyer asks a question in a web app and reads the answer as it is written | Online web service, answer streamed token by token | Time to first token under about 2 s and p95 of the whole answer under about 5 s (*proposed*): the person is waiting |
| (b) Every night: re-embed the corpus after an update, re-run the evaluation questions, and answer an archive of saved questions | Batch | Finished before the working day starts; the number that matters is cost per 1,000 answers, not latency |
| (c) Messages arrive from a law firm's inbox and each is tagged with the likely articles within seconds | Streaming (Redis Streams, consumer group) | Tag visible a few seconds after arrival (*proposed*); at-least-once delivery is fine because tagging the same message twice is harmless |

### Serving vocabulary (rewrite in your own words)

- **Latency vs throughput:** latency is how long one request takes; throughput is how many requests per second the service finishes. Batching raises throughput and costs latency.
- **p50 / p95 / p99:** the time under which 50 %, 95 % and 99 % of requests finish. The mean hides the slow tail that real users feel; p95 and p99 show it.
- **Cold start:** the time before the first request can be answered: here, downloading the registered version, loading the ONNX encoder and the vectors.
- **GPU utilization:** the share of time the GPU is busy. Low utilization under load means the GPU waits for the CPU or for batching.
- **Batch size:** how many inputs go through the model in one call. It is 1 in the current `/ask`, which wastes the model's ability to process several inputs at almost the same cost.

### The four-layer serving stack, for this system

The handbook PDF does not list the four layers, so check the names against Lesson 4.

```
1. Interface       client / curl / Locust ──► nginx (canary, Step 9) ──► /ask (stream), /health, /model
2. Serving         BentoML service: async endpoint, runner for the encoder, micro-batching, /metrics
3. Inference       ONNX Runtime (encoder, CPU) · vLLM or hosted Groq (generation)
4. Infrastructure  Docker containers on WSL · MLflow registry + RustFS (model versions) · Redis, Airflow
```

## Plan: the handbook steps applied to this project

| Step | Handbook | This project | Open decision |
|---|---|---|---|
| 1 | Airflow DAG extract → train → evaluate → branch → register | extract (new corpus) → validate → embed and evaluate (hit@3 on the dev questions) → branch with `registry challenge --target staging` → register → notify | none |
| 2 | Written pattern decision | this file | confirm the *proposed* SLAs |
| 3 | FastAPI baseline, four problems with evidence | batch size 1 (one query vs a batch of queries through the encoder), GIL (CPU plateau under load), eager vs ONNX (already measured), versioning (already solved by the registry, so explain that) | how to keep Groq out of the measurement: a stub LLM with fixed latency, or measure retrieval-only |
| 4 | BentoML with a Runner and micro-batching | async `/ask` with streaming; the encoder in a Runner with micro-batching | none |
| 5 | Batch scoring of at least 1 M rows | the RAG track asks for a batch re-indexing script tested with one new document; 1 M rows does not fit a 1,149-article corpus | confirm with the instructor what replaces 1 M rows |
| 6 | Redis Streams consumer | consumer group that tags incoming questions with articles, XACK, dead-letter stream | none |
| 7 | 7b ONNX Runtime + OpenVINO (required), plus 7a TensorRT or 7c vLLM | OpenVINO for the encoder; vLLM for generation (required by the RAG track) | GPU: local, or Colab / Kaggle documented as such |
| 8 | Locust to 100 users, bottleneck, one change | same; the RAG track asks for a report at 50 users in `reports/` | where Locust runs: not on the machine that serves |
| 9 | nginx canary with automatic rollback, plus shadow mode | two containers from two registry versions (`@production` and version 2), attributed by `/model`; shadow mode compares retrieved articles offline | none |

## Step 3 — FastAPI baseline (CAT 1)

Conditions for every load-test row in this report: Locust on the same laptop as the service (a limit: the machine is shared), the API container limited to 2 CPUs and 2 GB, the stub LLM answering after a fixed 800 ms (so the test measures this service and not the provider's rate limits), 80 % `/ask` with 3 articles, 15 % `/ask` with 10 articles, 5 % `/metadata`, ramp of 2 users per second. The laptop varies by about 20 % between identical runs (two 150-user rounds below), so differences of a few percent are not findings.

| Run | requests | failures | p50 (ms) | p95 (ms) | p99 (ms) | RPS |
|---|---|---|---|---|---|---|
| 50 users, 6 min | 5,920 | 0 | 900 | 1,300 | 1,500 | 16.5 |
| 150 users, 4 min, round 1 | 4,909 | 0 | 5,000 | 6,300 | 6,900 (`/ask k=3`) | about 20.5 |
| 150 users, 4 min, round 2 | 3,792 | 0 | 7,100 | 8,800 | 9,600 | 15.9 |

- At 50 users the latency is the stub's 800 ms plus about 100 ms for this service. Throughput there is limited by the users, not by the service: 50 users each waiting 1–3 s between questions and 0.9 s for the answer give at most about 17 requests/s, and 16.5 were measured.
- At 150 users the service is saturated: throughput barely moved (16–20 requests/s) while the median grew from 0.9 s to 5–7 s. The queue is shared: `/metadata`, which does no work, took 3.4 s at the median in round 1.
- Resources: at 50 users the container used 139 % CPU on average (peak 208 %) and 865 MiB; at 150 users 186 % on average, 197 % at the median, 716 MiB. The CPU limit is 200 %.

### The four problems, with evidence

| Problem | Evidence | Verdict |
|---|---|---|
| Batch size is 1 | Sweep over batch sizes (100 questions, ONNX Runtime CPU, median of 5): as the questions arrive 26.4 ms per question at batch 1, 17.0 at 2, **16.1 at 4**, 24.9 at 8, 20.4 at 16, 23.9 at 32, 29.3 at 100. Grouping questions by length first gives 18.8 at batch 1 and 15.6 at 4 (`reports/baseline-batching.md`) | **Partly supported**: small batches (2–4) are faster, larger ones are not. The batch-1 row is noisy (the same work reads 26.4 ms in arrival order and 18.8 ms grouped, and an earlier run of 100 single calls gave 18.0 ms), so the gain of batch 4 is between about 1.2x and 1.6x, not a precise number |
| The GIL caps the CPU | The container used up to 197 % CPU (median) with latency climbing, more than one core, so the process is not stuck on one core. The CPU limit is what is saturated | **Not supported**: the bottleneck is CPU capacity, not the GIL |
| Eager runtime | The service already runs ONNX Runtime. Module 1 measured 55.3 ms (ONNX) against 60.6 ms (PyTorch eager), about 9 % | Small. OpenVINO is tested in Step 7 |
| No model versioning | One image with the index baked in: a bad index or encoder means rebuilding and redeploying everything | **Solved in Module 2**: the retriever is a registry version (`@production`), switched by an alias change and a restart |

## Step 4 — BentoML (CAT 2)

Two services in two worker processes (`serving/service.py`): the `Encoder` (ONNX, with adaptive micro-batching available) and the API (`async` `/ask` and token-streaming `/ask_stream`, built-in `/livez`, `/readyz` and `/metrics`). In BentoML 1.4 the Runner of the handbook is a separate service reached with `bentoml.depends`. Image: `legal-rag-bento:0.3.0`.

### Results (same conditions as Step 3)

| Run | requests | failures | p50 (ms) | p95 (ms) | p99 (ms) | RPS |
|---|---|---|---|---|---|---|
| FastAPI, 50 users | 5,920 | 0 | 900 | 1,300 | 1,500 | 16.5 |
| Bento, batching off, 50 users | 5,832 | 0 | 930 | 1,500 | 2,300 | 16.3 |
| Bento, batch 2 / 1000 ms | 5,859 | 71 (1.2 %) | 960 | 1,300 | 1,600 | 16.3 |
| Bento, batch 32 / 2000 ms | 5,771 | 21 (0.4 %) | 940 | 1,700 | 2,900 | 16.1 |
| Bento, batch 16 / 1000 ms | 5,633 | 317 (5.6 %) | 990 | 1,800 | 2,300 | 15.7 |
| FastAPI, 150 users (round 2) | 3,792 | 0 | 7,100 | 8,800 | 9,600 | 15.9 |
| Bento, batching off, 150 users, before the fix below | 3,023 | 747 (24.7 %) | 9,500 | 14,000 | 16,000 | 12.7 |
| same, repeated (log kept) | 2,917 | 792 (27.2 %) | 9,800 | 17,000 | 19,000 | 12.2 |
| **Bento, batching off, 150 users, after the fix** | 4,470 | 0 | 6,100 | 6,800 | 7,200 | 18.7 |
| Bento, batch 16 / 3000 ms, 150 users, after the fix | 3,204 | 0 | 9,000 | 10,000 | 10,000 | 13.5 |

### Acceptance check: not met

The handbook asks for a measurable throughput improvement over CAT 1 at an equal or lower p95. BentoML did not give it.

- 50 users: same throughput (16.3 against 16.5 requests/s), higher p95 (1,500 against 1,300 ms).
- 150 users, after the fix: 18.7 requests/s, p95 6.8 s, no failures. FastAPI measured 20.5 requests/s with p95 6.3 s in one round and 15.9 requests/s with p95 8.8 s in the other. BentoML sits inside that range: with a laptop that varies by about 20 % between identical runs, the two are not distinguishable. No improvement, and no loss.
- Resources at 150 users: Bento 183 % CPU on average (peak 198 %) and 926 MiB; FastAPI 186 % (peak 250 %) and 716 MiB. Both are at the 200 % CPU limit.

### Mechanism, point by point

- **Separate process (Runner).** The point is to take inference out from under the API's GIL. Here the GIL was not the bottleneck (Step 3), so there was nothing to gain, and the price is an extra hop per question (serialization and an HTTP call between the two services) on the same two cores. After the fix, at 150 users, the container used 183 % CPU on average and 926 MiB against 186 % and 716 MiB for FastAPI, for 4,470 requests against 3,792 and 4,904 in the two FastAPI rounds: about 0.10 core-seconds per request against about 0.12 in the round that has CPU samples. That is the same order and inside the variation between runs, so the extra hop between the two processes did not show up as a measurable cost, and the second process (about 200 MiB more memory) bought no measurable throughput.
- **Micro-batching.** The sweep shows small batches (2–4) a little faster than single questions and large ones not (batch 100 was 0.9x), so there is little to collect: the encoder is about 16–26 ms of the roughly 100 ms of CPU each request costs at 150 users (183 % CPU x 240 s / 4,470 requests, an estimate), so even a 1.3x gain on the encoder is a gain of a few percent on the whole request, below the run-to-run variation. Measured at 150 users, batch 16 with 3000 ms was clearly worse than no batching (13.5 against 18.7 requests/s, p50 9.0 s against 6.1 s): a batch waits for company and its slowest member. `max_latency_ms` is the latency the encoder step is allowed to take, queueing included: when BentoML predicts a request cannot finish within it, it answers 503 ("process is overloaded"). With batching off there were 0 such answers in 5,832 requests at 50 users; with batching, 0.4 % to 5.6 % depending on the two numbers, and a value of 10 ms rejected requests on a healthy service. A batch size of 4 was not run at 150 users.
- **Runtime.** Unchanged (ONNX Runtime in both), so no difference is expected from it.
- **What BentoML does give:** an `async` `/ask` (the wait for the LLM is an `await`, not a blocked thread), token streaming, and liveness, readiness and metrics endpoints without code. On 50 users its p50 (930 ms) and throughput (16.3 requests/s) are within the laptop's run-to-run variation of FastAPI's; its p95 and p99 are higher.
- **Failures at 150 users, and their cause.** All of them are HTTP 500 from `/ask` (619 of `k=3`, 128 of `k=10` in the first run; 669 and 123 in the repeat), starting with the ramp and continuing to the end. The server log of the repeat shows 1,553 `ServerDisconnectedError` raised by BentoML's client between the API and the Encoder, plus a `Connector is closed` and a `Connection closed`, and no error at all from the Encoder, which kept answering. 1,553 is about twice the 792 failed requests: each failed request had the first call and its single retry fail. The cause is in my service, not in the model or the laptop: at 150 users the API opened about one connection per user to the Encoder, and under that load BentoML's inter-service client drops connections.
- **Fix.** Cap the number of calls waiting on the Encoder at once (`BENTO_ENCODER_CONCURRENCY`, default 16; the rest wait in the API's event loop, which costs nothing) and retry up to twice (`BENTO_ENCODER_RETRIES`). I reproduced the same exceptions on a test machine (1 CPU, an encoder that burns 18 ms of CPU per question, 150 users, 60 s) and measured each protection alone and together:

  | Protections | failures | p50 (ms) | p95 (ms) |
  |---|---|---|---|
  | one retry, no cap (the version that failed) | 61.7 % | 1,300 | 5,100 |
  | two retries, no cap | 10.8 % | 3,000 | 6,700 |
  | cap of 16, no retry | 11.5 % | 1,300 | 1,600 |
  | cap of 16 and two retries | 0 % | 1,700 | 2,100 |

  Those are numbers from a test machine, not from the laptop; the fix is confirmed only when the 150-user run on the laptop is repeated.

### Streaming

`/ask_stream` sends the answer while it is written and the sources last. With the stub LLM (800 ms) the first piece arrived after 0.28 s and the answer finished after 0.86 s, in 13 pieces (`reports/stream-demo.txt`, `scripts/stream_demo.py`). The 0.28 s is a property of the stub, which emits its first word after 30 % of its delay; the time to first token with the real LLM was not measured here.

### Limits of these measurements

One run per setting, Locust on the same machine as the service, a stub LLM with a fixed delay (so the cost of a long prompt is not measured: `k=10` was as fast as `k=3`), and a laptop that varies by about 20 % between identical runs.

## Remaining

Steps 1, 5, 6, 7, 8 and 9, the final reading of Step 2's definitions in your own words, and pushing the fixed image (`doaamostafa679/legal-rag-bento:0.3.0`) — _pending_
