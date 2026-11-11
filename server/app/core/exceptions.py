"""Domain exceptions and the consistent error response shape used by the API."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    message: str


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class UnsupportedDocumentError(AppError):
    status_code = 415
    code = "unsupported_document"


class ExtractionError(AppError):
    """A file could not be parsed (corrupt, encrypted, empty...)."""

    status_code = 422
    code = "extraction_failed"


class ScannedDocumentError(ExtractionError):
    """A PDF with no usable text layer (a scan). It is left out of the run, not treated as a failure,
    because the system has no OCR."""

    code = "scanned_document"


def register_exception_handlers(app: FastAPI) -> None:
    from app.llm.errors import LLMConfigError, LLMError

    @app.exception_handler(LLMConfigError)
    async def _llm_config(_: Request, exc: LLMConfigError) -> JSONResponse:
        return JSONResponse(status_code=503, content=ErrorResponse(error="llm_not_configured", message=str(exc)).model_dump())

    @app.exception_handler(LLMError)
    async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=502, content=ErrorResponse(error="llm_error", message=str(exc)).model_dump())

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=exc.code, message=exc.message).model_dump(),
        )
