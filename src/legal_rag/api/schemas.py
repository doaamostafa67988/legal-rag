"""Request/response models for the HTTP API."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"question": "ما هو سن الرشد؟", "k": 3}}
    )

    question: str = Field(min_length=1, max_length=1000, description="Arabic or English")
    k: int = Field(default=3, ge=1, le=10, description="How many articles to retrieve")

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class AskResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "سن الرشد 21 سنة ميلادية كاملة (المادة 44).",
                "sources": ["Egyptian Civil Code, Art. 44"],
                "correlation_id": "0f8c2d4e5b6a4c7d8e9f0a1b2c3d4e5f",
                "latency_ms": 840.2,
            }
        }
    )

    answer: str
    sources: list[str] = Field(description="Article citations, never chunk ids")
    correlation_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    embedder: str
