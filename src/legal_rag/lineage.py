"""Link an MLflow run to the DVC versions of the data it used (Module 2, Step 4).

`dvc repro` records, in `dvc.lock`, the md5 of the PDF that went into the `parse` stage and of
the corpus JSON that came out. Every run logs those two hashes as tags, so from a run you can
`git checkout` the commit and `dvc pull` the exact data. `dvc_in_sync` says whether the corpus
file on disk is the one the lock describes: "false" means the run used a corpus nobody can
reproduce (edited by hand, or `dvc repro` not run after the parser changed).

`data_version` (first 12 characters of the corpus SHA-256) stays as it was, so runs logged
before DVC remain comparable; the DVC hashes are extra tags, not a replacement.
"""

import hashlib
from pathlib import Path

import yaml

from legal_rag.config import ROOT, settings

LOCK = ROOT / "dvc.lock"
STAGE = "parse"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def lock_hashes(lock_path: Path = LOCK, stage: str = STAGE) -> dict[str, str]:
    """md5 of the stage's PDF input and JSON output as recorded in dvc.lock ({} without one)."""
    if not lock_path.is_file():
        return {}
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    entry = (lock.get("stages") or {}).get(stage) or {}
    found: dict[str, str] = {}
    for dep in entry.get("deps") or []:
        if str(dep.get("path", "")).endswith(".pdf"):
            found["source_pdf_md5"] = dep["md5"]
    for out in entry.get("outs") or []:
        if str(out.get("path", "")).endswith(".json"):
            found["corpus_md5"] = out["md5"]
    return found


def data_lineage(lock_path: Path = LOCK, corpus_json: Path | None = None) -> dict[str, str]:
    """MLflow tags tying a run to its DVC data; empty until the project has a dvc.lock."""
    hashes = lock_hashes(lock_path)
    if not hashes:
        return {}
    tags = {f"dvc_{key}": value for key, value in hashes.items()}
    if "corpus_md5" in hashes:
        corpus = corpus_json or settings.corpus_json
        in_sync = corpus.is_file() and _md5(corpus) == hashes["corpus_md5"]
        tags["dvc_in_sync"] = str(in_sync).lower()
    return tags
