"""FastAPI service: POST /ask, GET /health.

The embedder and the Chroma index load once at startup (lifespan), never per request. Every
request gets a correlation id (X-Request-ID) that appears in all of its log lines.
"""

import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from legal_rag.api.schemas import AskRequest, AskResponse, HealthResponse
from legal_rag.config import settings
from legal_rag.generate import ChatClient, GroqChat, answer
from legal_rag.logging_conf import configure_logging, correlation_context, correlation_id_var
from legal_rag.rag import LegalRAG, sources_of

log = logging.getLogger(__name__)

_VALID_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


class CorrelationIdMiddleware:
    """Pure ASGI middleware: set the correlation id, log the request, echo X-Request-ID.

    It also turns unexpected errors into a clean 500 here, inside the correlation context, so
    the traceback is logged with the request's id and never reaches the client.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        start = time.perf_counter()
        status = 500
        started = False
        with correlation_context(incoming if _VALID_ID.fullmatch(incoming) else None) as cid:

            async def send_with_id(message: Message) -> None:
                nonlocal status, started
                if message["type"] == "http.response.start":
                    started = True
                    status = message["status"]
                    headers = [*message.get("headers", []), (b"x-request-id", cid.encode())]
                    message["headers"] = headers
                await send(message)

            try:
                await self.app(scope, receive, send_with_id)
            except Exception:
                log.exception("unhandled error", extra={"path": scope["path"]})
                if not started:
                    body = {"detail": "Internal server error", "correlation_id": cid}
                    await JSONResponse(body, status_code=500)(scope, receive, send_with_id)
            log.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )


def _load_rag() -> LegalRAG:
    if settings.embedder_backend == "onnx":
        from legal_rag.onnx_embed import OnnxEmbedder

        return LegalRAG(embedder=OnnxEmbedder())
    return LegalRAG()


def create_app(rag: LegalRAG | None = None, chat: ChatClient | None = None) -> FastAPI:
    """Build the app. Tests pass fakes; production loads the real ones at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        try:
            app.state.rag = rag or _load_rag()
            app.state.chat = chat or GroqChat()
        except Exception:
            log.exception("startup failed: model or index could not be loaded")
            raise
        log.info("service ready", extra={"documents_indexed": app.state.rag.documents_indexed})
        yield

    app = FastAPI(title="Egyptian Civil Code Q&A", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CorrelationIdMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
            for e in exc.errors()
        ]
        log.warning("request rejected", extra={"path": request.url.path, "errors": errors})
        return JSONResponse({"detail": errors}, status_code=422)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> JSONResponse | HealthResponse:
        rag_ready = getattr(request.app.state, "rag", None)
        if rag_ready is None:
            return JSONResponse({"detail": "model not loaded"}, status_code=503)
        return HealthResponse(
            status="ok",
            documents_indexed=rag_ready.documents_indexed,
            embedder=type(rag_ready.embedder).__name__,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        # plain `def`: FastAPI runs it in a worker thread (copying contextvars), so the
        # blocking embed + LLM calls do not stall the event loop.
        start = time.perf_counter()
        hits = request.app.state.rag.retrieve(body.question, k=body.k)
        text = answer(body.question, hits, chat=request.app.state.chat)
        return AskResponse(
            answer=text,
            sources=sources_of(hits),
            correlation_id=correlation_id_var.get(),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    return app


app = create_app()
