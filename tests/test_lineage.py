import hashlib

from legal_rag.lineage import data_lineage, lock_hashes

LOCK = """\
schema: '2.0'
stages:
  parse:
    cmd: python -m legal_rag.parse_civil_code
    deps:
    - path: data/raw/laws.pdf
      hash: md5
      md5: aaaa1111
      size: 10
    - path: src/legal_rag/parse_civil_code.py
      hash: md5
      md5: cccc3333
      size: 5
    outs:
    - path: data/processed/civil_code.json
      hash: md5
      md5: {corpus_md5}
      size: 20
"""


def write_lock(tmp_path, corpus_md5="bbbb2222"):
    lock = tmp_path / "dvc.lock"
    lock.write_text(LOCK.format(corpus_md5=corpus_md5), encoding="utf-8")
    return lock


def test_no_lock_means_no_extra_tags(tmp_path):
    assert lock_hashes(tmp_path / "dvc.lock") == {}
    assert data_lineage(tmp_path / "dvc.lock") == {}


def test_reads_the_pdf_and_corpus_hashes_and_ignores_the_script(tmp_path):
    assert lock_hashes(write_lock(tmp_path)) == {
        "source_pdf_md5": "aaaa1111",
        "corpus_md5": "bbbb2222",
    }


def test_other_stage_names_are_ignored(tmp_path):
    assert lock_hashes(write_lock(tmp_path), stage="train") == {}


def test_in_sync_when_the_corpus_matches_the_lock(tmp_path):
    corpus = tmp_path / "civil_code.json"
    corpus.write_bytes(b"[]")
    md5 = hashlib.md5(b"[]", usedforsecurity=False).hexdigest()
    tags = data_lineage(write_lock(tmp_path, md5), corpus)
    assert tags["dvc_in_sync"] == "true" and tags["dvc_corpus_md5"] == md5
    assert tags["dvc_source_pdf_md5"] == "aaaa1111"


def test_out_of_sync_when_the_corpus_was_edited(tmp_path):
    corpus = tmp_path / "civil_code.json"
    corpus.write_bytes(b"[1]")
    assert data_lineage(write_lock(tmp_path), corpus)["dvc_in_sync"] == "false"


def test_out_of_sync_when_the_corpus_is_missing(tmp_path):
    assert data_lineage(write_lock(tmp_path), tmp_path / "gone.json")["dvc_in_sync"] == "false"
