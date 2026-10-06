# legal-rag

Bilingual (AR/EN) Egyptian Civil Code RAG: parent-child retrieval, reranking and GraphRAG.

## Setup
```bash
uv sync
uv run pre-commit install
```

## Build the corpus
Put the source PDF at `data/raw/laws.pdf`, then:
```bash
uv run python -m legal_rag.parse_civil_code data/raw/laws.pdf data/processed/civil_code.json
```

## Test
```bash
uv run pytest -q
```
