from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEGAL_RAG_", env_file=".env")

    raw_pdf: Path = ROOT / "data" / "raw" / "laws.pdf"
    corpus_json: Path = ROOT / "data" / "processed" / "civil_code.json"
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
