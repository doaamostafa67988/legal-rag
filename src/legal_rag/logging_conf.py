"""Structured JSON logging with a per-request correlation id (stdlib only).

Every line is one JSON object: timestamp, level, logger, message, correlation_id, plus any
``extra={...}`` fields and the traceback when there is one. The correlation id lives in a
ContextVar, so concurrent requests (threads or asyncio tasks) never see each other's id.
"""

import contextvars
import json
import logging
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)

# Attributes every LogRecord has; anything else on a record came from `extra=`.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
# Libraries that log every HTTP request at INFO; keep them to warnings and above.
_NOISY = ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3", "filelock")


def new_correlation_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def correlation_context(correlation_id: str | None = None) -> Iterator[str]:
    """Tag every log line emitted inside the block with one correlation id."""
    cid = correlation_id or new_correlation_id()
    token = correlation_id_var.set(cid)
    try:
        yield cid
    finally:
        correlation_id_var.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.simplefilter("default")
    logging.captureWarnings(True)  # Python warnings become JSON lines too
