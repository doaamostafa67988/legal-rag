"""Paths must not depend on where the package is installed."""

import os
import subprocess
import sys

from legal_rag.config import settings

PRINT_PATHS = (
    "from legal_rag.config import settings as s; "
    "print(s.chroma_dir); print(s.onnx_dir); print(s.corpus_json)"
)


def test_default_paths_point_into_the_checkout():
    assert settings.corpus_json.is_file()


def test_legal_rag_root_moves_every_data_path(tmp_path):
    env = {**os.environ, "LEGAL_RAG_ROOT": str(tmp_path)}
    out = subprocess.run(
        [sys.executable, "-c", PRINT_PATHS], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == [
        str(tmp_path / "chroma_db"),
        str(tmp_path / "models" / "e5-small-onnx"),
        str(tmp_path / "data" / "processed" / "civil_code.json"),
    ]


def test_a_single_path_can_still_be_overridden(tmp_path):
    env = {**os.environ, "LEGAL_RAG_ROOT": str(tmp_path), "LEGAL_RAG_CHROMA_DIR": "/data/index"}
    out = subprocess.run(
        [sys.executable, "-c", PRINT_PATHS], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[0] == "/data/index"
