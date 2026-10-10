# syntax=docker/dockerfile:1
# Build from the repo root:  docker build -f docker/Dockerfile -t legal-rag:0.1.0 .

# ---- builder: has uv and the build cache, none of which ship ------------------------------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# Same path as in the runtime stage: a virtualenv is not relocatable.
WORKDIR /app

# 1) third-party dependencies only: this layer is cached until pyproject.toml / uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) the package itself, installed (not editable) into the same venv
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable


# ---- runtime: python + the venv + the data the service needs ------------------------------
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
# The corpus, the exported ONNX encoder and the Chroma index travel inside the image.
COPY --chown=appuser:appuser data/processed/civil_code.json data/processed/civil_code.json
COPY --chown=appuser:appuser models/e5-small-onnx/ models/e5-small-onnx/
COPY --chown=appuser:appuser chroma_db/ chroma_db/

ENV PATH="/app/.venv/bin:$PATH" \
    LEGAL_RAG_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False
# GROQ_API_KEY is deliberately NOT set here: pass it at run time (-e GROQ_API_KEY=...).

USER appuser
EXPOSE 8000
# the model takes a while to load, so give it a start period before failures count
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# the app already logs one JSON line per request, so uvicorn's access log is switched off
CMD ["uvicorn", "legal_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
