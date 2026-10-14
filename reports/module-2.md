# Module 2 — Tracking, versioning, automation

Branch: `module-2-tracking-automation` · Tag after merge: `v0.2.0`

Status: Steps 1–2 done (this file). Steps 3–7 are listed at the bottom and filled in as they are finished.

## Step 1 — Tracking infrastructure

`docker/docker-compose.tracking.yml` runs three services: PostgreSQL (run metadata), RustFS, an S3-compatible object store (artifacts), and the MLflow 3.17.0 server.

```bash
docker compose -f docker/docker-compose.tracking.yml --env-file .env up -d --build
# MLflow UI: http://localhost:5000      object-store console: http://localhost:9001
```

`MLFLOW_TRACKING_URI` is read through `legal_rag.config` (default `http://localhost:5000`). The API compose file stays separate, so `docker compose up` for the API never starts the tracking stack.

## Step 2 — Experiment tracking

### What is tracked

Every run logs:

- **Params:** `embedding_model`, `variants`, `fusion`, `weights`, `chunk_strategy`, `chunk_size`, `overlap`, `corpus_sha256`, `eval_set_sha256`, `n_docs`, `n_questions`, `split`. Finalist runs add `faithfulness_judge`, `faithfulness_lang`, `faithfulness_n`, `faithfulness_gate`.
- **Metrics:** `hit_at_3`, `mrr` (overall and per language), `n_vectors`, `index_mb`; finalists also `faithfulness`, `faithfulness_below_gate`, `citation_valid_rate`, `judge_tokens_cold`.
- **Artifacts:** `misses.json` (questions the config failed), `requirements.txt` (the locked environment from `uv export`); finalists also `faithfulness.json`.
- **Tags:** `git_commit`, `data_version` (first 12 characters of the corpus SHA-256), `author`, `framework`, `family`.

Because the corpus is chunked by article, one article is one chunk, so `chunk_size = "article"` and `overlap = 0` are logged as constants. What varies between runs is what is embedded for an article (Arabic, English, or both) and how the scores are fused.

### Three experiments, on purpose

| Experiment | Question it answers | Code |
|---|---|---|
| `legal-rag-retrieval` | Which representation retrieves best? Full sweep: 3 single-vector variants, a max fusion and an 11-step weighted fusion (nested runs), per embedding model | `experiments.py` |
| `legal-rag-finalists` | Of the best configs, which also gives faithful answers? Retrieval and faithfulness side by side | `finalists.py`, `faithfulness.py` |
| `legal-rag-holdout` | Does the choice survive questions it was never tuned on? Run once, never used to tune | `experiments.py --split holdout` |

They are separate because they answer different questions, and the hold-out must stay apart from the dev runs. The comparison view for this report is `legal-rag-finalists`, the only experiment where every run has both retrieval metrics and `faithfulness`. All finalist runs share `git_commit = 11ec18a` and one `data_version`, so the comparison is on the same code and the same data. `finalists.py`, which produced these runs, was committed afterwards (`a4b9064`), so that tag points at the code without it.

### Autologging

Autologging is not applicable here: nothing is trained. The embedding models are used as they are, and the retrieval code is plain numpy. Everything above is logged manually, which is also why the metrics and the data hash are explicit.

### Comparison on the dev set (30 questions; faithfulness on 15 Arabic questions)

| Run | hit@3 | MRR | Arabic hit@3 | Faithfulness | Index (MB) |
|---|---|---|---|---|---|
| e5-base / ar and en vectors, max | 1.000 | 0.922 | 1.000 | 1.000 | 6.72 |
| e5-small / ar and en vectors, max | 1.000 | 0.817 | 1.000 | 0.983 | 3.36 |
| e5-base / alpha=0.5 | 0.967 | 0.861 | 0.933 | 0.962 | 6.72 |
| e5-base / ar+en | 0.933 | 0.834 | 0.933 | 0.942 | 3.36 |
| e5-base / ar | 0.733 | 0.642 | 1.000 | 1.000 | 3.36 |
| e5-small / ar | 0.700 | 0.557 | 1.000 | 0.983 | 1.68 |

![MLflow comparison view, legal-rag-finalists](module-2-finalists.png)

How to read it:

- Mixing Arabic and English vectors (`max`) is what lifts hit@3 from 0.70–0.73 to 1.0 on the dev set.
- Faithfulness does not separate the configs: it ranges from 0.94 to 1.0 on 15 questions. It measures whether the answer is supported by the retrieved articles, not whether the right articles were retrieved (`e5-base / ar` has faithfulness 1.0 and hit@3 0.73). So hit@3 and MRR choose the config, and faithfulness stays as a gate against unsupported answers.
- `e5-small / max` reaches the same hit@3 as `e5-base / max` with half the index (3.36 MB against 6.72 MB), at a lower MRR (0.82 against 0.92).
- One dev question is 0.033 of hit@3, so differences of one question are signal, not proof. The config was also chosen on these same questions, which is why the hold-out exists.

## Hold-out check (20 questions, never used for tuning)

| Run | hit@3 | MRR |
|---|---|---|
| e5-base / ar and en vectors, max | 0.650 | 0.608 |
| e5-small / ar and en vectors, max | 0.600 | 0.524 |
| e5-small / ar | 0.500 | 0.429 |

- **The ranking holds:** mixing both languages beats Arabic alone (+0.10 hit@3) on unseen questions too.
- **The level does not:** `e5-small / max` scored 1.0 on dev and 0.6 on the hold-out. The honest expectation for this config on new questions is hit@3 around 0.6, not 1.0.
- **Base against small is inside the noise:** 13 against 12 questions out of 20. The base model fixed two questions (h10, h14) and broke one (h19). Its index is twice as large and serving it would need an ONNX export of e5-base.
- **The hold-out was looked at twice:** once for e5-small, then once for e5-base (run at commit `a4b9064`), so it is no longer a completely clean test. Both runs are in the `legal-rag-holdout` experiment.

### Why questions fail (`misses.json` of the hold-out runs)

Five of the 20 questions fail in all three runs (h07, h08, h12, h15, h18). The failures fall into three patterns:

1. **Near misses, right area, wrong article.** h07 (voidability under duress): gold is article 127, the retrievers return 125, 126, 138 or 140. h02 (a person's domicile): gold 40, retrieved 41 and 42. h19 (abuse of ownership): gold 807, retrieved 805 and 818. Neighbouring articles are probably relevant to a lawyer, so hit@3 against a single gold article is strict for these questions.
2. **Wrong legal topic.** h15 (employer liability for an employee's harm): gold is 174 (tort liability), every run returns articles in the 665–692 range, which are about employment contracts. The words "employer" and "worker" pull retrieval to the wrong part of the code. A re-ranker is the usual fix.
3. **English questions on an Arabic-only index.** 8 of the 10 failures of `e5-small / ar` are English questions; `max` reduces them to 6 (small) and 4 (base).

The misses were read to understand failure types only. The pipeline is not tuned to these 20 questions; any improvement is checked on new questions.

### Decision

- **The rule, committed before the e5-base hold-out run** (commit `b60f2d5`, section "Decision rule, written before the second hold-out look"): adopt e5-base only if its hold-out hit@3 is higher than 0.600 (at least 13 of 20); a tie or a lower score keeps e5-small.
- **The result:** e5-base scored 0.650 (13 of 20). By that rule, e5-base should have been adopted.
- **What we decided instead: we deviate from the rule and register e5-small / ar and en vectors, max.** The reasons:
  1. The gain is one question (13 against 12 of 20), inside the noise of a 20-question set.
  2. Cost. The base index is twice as large (6.72 MB against 3.36 MB, measured). The base model is also much larger than the small one (the e5-small ONNX export is 470 MB, measured; e5-base is roughly 1.1 GB, an estimate) and would need its own ONNX export and a slower query. The API already serves e5-small.
- **What this decision does not rest on:** the query latency and the ONNX size of e5-base were not measured, so the cost argument is an estimate. MRR was higher for base on both sets (+0.105 on dev, +0.084 on the hold-out), so its gain may be real but small. The rule would have chosen base; choosing small is a judgment call, recorded as a deviation.
- **When to revisit:** if new questions (not the current hold-out) show a clear gain for base, switching is a one-line alias change in the registry.

## Problems met

- **Artifact download hung** (`misses.json` could not be downloaded, retries every 90 s). The MLflow server answered 200 on every request, but the presigned download URL it returns points at `http://s3:9000/...`, the container name inside the Compose network. The host (WSL) cannot resolve `s3`. Uploading worked because it goes through the server. Fix: `127.0.0.1 s3` in `/etc/hosts` (port 9000 is published on localhost, and the signature stays valid because the `Host` header stays `s3:9000`). WSL regenerates `/etc/hosts` on restart; `generateHosts = false` in `/etc/wsl.conf` keeps it.
- **Unpinned image:** `rustfs/rustfs:latest` in the tracking compose file; to be pinned to a digest.

## Remaining steps

- **Step 3 — Model registry:** register the retrieval config, alias `production`, load it by alias in the API, `promote_if_better` in `registry.py` — _pending_
- **Step 4 — DVC:** source PDF and `civil_code.json` tracked, `dvc.yaml` stage for the parser, remote, lineage tag in MLflow — _pending_
- **Step 5 — CI:** lint → test → faithfulness gate (< 0.75 fails, 20 questions) → build → push; branch protection; red/green screenshots — _pending_
- **Step 6 — Terraform:** Docker provider for Postgres + RustFS + MLflow; destroy and apply — _pending_
- **Step 7 — Continuous training:** schedule, manual and `repository_dispatch` triggers; Staging automatic, Production behind a manual approval; CT decision table — _pending_
