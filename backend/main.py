"""FastAPI application entry point for ISL Bridge."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes_health import router as health_router
from backend.api.routes_predict import router as predict_router
from backend.api.routes_sign import get_sign_retriever, router as sign_router
from backend.core.config import get_settings
from backend.core.constants import SERVICE_NAME
from backend.services.inference_service import get_inference_service

logger = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    """Configure application-wide logging using the configured log level."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load and cache the inference model once when the backend starts."""
    settings = get_settings()
    application.state.sign_retriever = get_sign_retriever()
    service = get_inference_service()
    application.state.inference_service = service
    logger.info(
        "ISL Bridge Backend | Environment=%s | Device=%s | Model=CNN-BiLSTM | Classes=%s | Input dimension=%s | Max frames=%s | Checkpoint=%s",
        settings.environment,
        service.device,
        service.num_classes,
        service.input_dim,
        service.max_frames,
        service.checkpoint_path,
    )
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Backend service for the ISL Bridge communication system.",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(Exception)
    async def handle_unexpected_exception(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Log unexpected errors and return a safe internal-server response."""
        logger.exception(
            "Unhandled exception while processing %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred.",
                "service": SERVICE_NAME,
            },
        )

    application.include_router(health_router)
    application.include_router(predict_router)
    application.include_router(sign_router)

    frontend_root = Path(__file__).resolve().parents[1] / "frontend"
    application.mount("/css", StaticFiles(directory=frontend_root / "css"), name="frontend-css")
    application.mount("/js", StaticFiles(directory=frontend_root / "js"), name="frontend-js")
    frontend_file = frontend_root / "index.html"

    @application.get("/", include_in_schema=False)
    async def frontend_index() -> FileResponse:
        return FileResponse(frontend_file, media_type="text/html")

    return application


app = create_app()
