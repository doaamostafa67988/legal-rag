"""Opt-in check with the real e5 model: `uv run pytest -m integration` (slow, downloads ~0.5 GB).

Reproduces the baseline notebook's evaluation, so the package must keep its hit@3.
"""

import pytest

from legal_rag.data import load_corpus
from legal_rag.rag import E5Embedder, LegalRAG, build_index

EVAL_SET = [
    ("ماذا يطبق القاضي إذا لم يوجد نص تشريعي؟", 1),
    ("What does the judge apply if there is no applicable legislative provision?", 1),
    ("هل يجوز لأحد المتعاقدين تعديل العقد بإرادته المنفردة؟", 147),
    ("Can one party unilaterally modify a contract?", 147),
    ("هل يجب تنفيذ العقد بحسن نية؟", 148),
    ("Must a contract be performed in good faith?", 148),
    ("ما التزام من يتسبب بخطئه في ضرر للغير؟", 163),
    ("What is the obligation of a person whose fault causes harm to another?", 163),
    ("ما هو سن الرشد؟", 44),
    ("What is the age of majority?", 44),
]


@pytest.mark.integration
def test_baseline_hit_at_3_matches_the_notebook(tmp_path):
    corpus = load_corpus()
    embedder = E5Embedder()
    build_index(corpus, embedder, chroma_dir=tmp_path)
    rag = LegalRAG(embedder=embedder, chroma_dir=tmp_path, articles=corpus)
    hits = sum(
        gold in {h.article_number for h in rag.retrieve(question, k=3)}
        for question, gold in EVAL_SET
    )
    assert hits / len(EVAL_SET) >= 0.8  # the notebook scored 0.9 (9/10)
