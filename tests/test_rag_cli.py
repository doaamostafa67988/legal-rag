"""The `python -m legal_rag.rag` CLI, with fakes so it never touches the real index or model."""

import pytest

from legal_rag import rag as rag_module


@pytest.fixture(autouse=True)
def _no_global_logging(monkeypatch):
    monkeypatch.setattr(rag_module, "configure_logging", lambda *a, **k: None)


def test_query_command_prints_scored_citations(monkeypatch, rag, capsys):
    monkeypatch.setattr(rag_module, "LegalRAG", lambda: rag)
    rag_module.main(["query", "ما هو سن الرشد", "-k", "1"])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert out[0].endswith("Egyptian Civil Code, Article 44")


def test_ingest_command_builds_the_index_from_the_corpus(monkeypatch, articles, embedder):
    built = {}
    monkeypatch.setattr(rag_module, "load_corpus", lambda: articles)
    monkeypatch.setattr(rag_module, "E5Embedder", lambda: embedder)
    monkeypatch.setattr(
        rag_module, "build_index", lambda arts, emb: built.update(articles=arts, embedder=emb)
    )
    rag_module.main(["ingest"])
    assert built == {"articles": articles, "embedder": embedder}


def test_an_unknown_command_exits_with_an_error():
    with pytest.raises(SystemExit):
        rag_module.main(["nope"])
