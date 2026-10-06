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