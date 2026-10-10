import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Where data/, chroma_db/ and models/ live. From a checkout that is the repo root; once the package
# is installed (Docker) __file__ sits inside site-packages, so deployments set LEGAL_RAG_ROOT.
ROOT = Path(os.environ.get("LEGAL_RAG_ROOT") or Path(__file__).resolve().parents[2])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEGAL_RAG_", env_file=ROOT / ".env", extra="ignore"
    )

    raw_pdf: Path = ROOT / "data" / "raw" / "laws.pdf"
    corpus_json: Path = ROOT / "data" / "processed" / "civil_code.json"
    chroma_dir: Path = ROOT / "chroma_db"
    onnx_dir: Path = ROOT / "models" / "e5-small-onnx"
    embedding_model: str = "intfloat/multilingual-e5-small"
    log_level: str = "INFO"
    embedder_backend: Literal["onnx", "torch"] = "onnx"  # what the API serves with
    groq_model: str = "qwen/qwen3.8-27b"
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    low_score_threshold: float = 0.75  # provisional: calibrate with off-topic queries
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
