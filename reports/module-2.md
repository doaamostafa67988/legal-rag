# Module 2 - Tracking and automation

## Step 02 - Tracking the experiments

- MLflow runs in Docker with PostgreSQL as the backend store and RustFS (S3-compatible) as the
  artifact store, instead of the MinIO named in the handbook. `MLFLOW_TRACKING_URI` comes from `.env`.
- Sweep on the 30 dev questions (15 Arabic + 15 English): 2 embedding models (e5-small, e5-base) x
  15 configs (single vector ar / en / ar+en, the best of the Arabic and English vectors, and a weighted
  mix with alpha 0 to 1). One chunk per article, so `chunk_size=article` and `overlap=0`.
- Every run records the corpus and eval-set SHA-256, the git commit, `requirements.txt`, the author
  and the size of the index.

### Six finalists, compared in one MLflow experiment (`legal-rag-finalists`)

Finalists were chosen on the dev questions only. `ar_*` columns use the 15 Arabic questions, the
others use all 30. Faithfulness is RAGAS on the same 15 Arabic questions (judge `openai/gpt-oss-120b`,
a different family from the generator).

| Run | faithfulness | ar hit@3 | ar MRR | hit@3 | MRR | index MB |
|---|---|---|---|---|---|---|
| e5-small / ar | 0.983 | 1.000 | 0.778 | 0.700 | 0.557 | 1.68 |
| e5-small / ar and en vectors, max | 0.983 | 1.000 | 0.778 | 1.000 | 0.817 | 3.36 |
| e5-base / ar | 1.000 | 1.000 | 0.878 | 0.733 | 0.642 | 3.36 |
| e5-base / ar+en | 0.942 | 0.933 | 0.833 | 0.933 | 0.834 | 3.36 |
| e5-base / alpha=0.5 | 0.962 | 0.933 | 0.789 | 0.967 | 0.861 | 6.72 |
| e5-base / ar and en vectors, max | 1.000 | 1.000 | 0.878 | 1.000 | 0.922 | 6.72 |

What the table says:

- The Arabic questions are saturated (ar hit@3 is 0.93 to 1.00 for every finalist). The English
  questions separate the configs: a single Arabic vector gets hit@3 0.70 to 0.73 over all 30
  questions, so only about 0.40 to 0.47 on the 15 English ones (derived: overall x 30 minus Arabic x
  15), while the two-vector `max` reaches 1.000.
- On the dev set the best config is e5-base / max (hit@3 1.000, MRR 0.922). Against e5-small / max the
  hit@3 is tied and the MRR is 0.105 higher.
- Faithfulness is 0.94 to 1.00 for all six, so it does not choose between them. It measures whether
  the answer is supported by the retrieved articles, not whether the right articles were retrieved.
  It works as a quality gate, not as a selection metric. Single scores move by 0.02 to 0.06 with one
  unsupported claim, so these differences are noise.
- The index size is small for every config. The real price of e5-base is the model: about 1.1 GB
  of ONNX weights against 470 MB for e5-small, and a slower query embedding. Neither is measured yet.

Validity of the faithfulness number:

- The first run scored 0.928. After the citation-stripping fix it is 0.983 (15 Arabic questions).
  The fix was made after seeing which questions scored low, so the dev number is optimistic. It
  corrects the scoring, it does not show a better system.
- Negative control (`legal_rag.faithfulness_control`, rule fixed before the run): the same answers
  scored against the articles of an unrelated question. Mean faithfulness fell from 0.98 to 0.01
  (14 of 15 scores were 0.00, the highest 0.17; n=15), so the judge notices an answer that is not
  grounded in its context. Subtle errors, such as one changed number in an otherwise correct answer,
  were not tested.

### Hold-out, first look (e5-small only)

The 20-question hold-out was checked against the dev set before any run: no article and no
question text in common (one hold-out question was replaced because its article, 125, was also in
the dev set). It was run once for two configs.

| Config | hold-out hit@3 | hold-out MRR |
|---|---|---|
| e5-small / ar | 0.500 | 0.429 |
| e5-small / ar and en vectors, max | 0.600 | 0.524 |

The dev score of the second config was 1.000, so the dev set overstated it by 0.40 (it was picked
as the best of 16 configs on 30 questions). The gain over the baseline is 2 questions out of 20,
inside the noise.

### Decision rule, written before the second hold-out look

- Candidate: e5-base / ar and en vectors, max. Incumbent: e5-small / ar and en vectors, max.
- The candidate is run once on the hold-out. It is adopted only if its hold-out hit@3 is higher than
  0.600 (at least 13 of 20). A tie or a lower score keeps e5-small, which is cheaper to serve.
- One question is 0.05 on 20 questions, so this is a decision rule, not a significance test.
  The hold-out will have been looked at in two sessions; this is stated in the final results.
