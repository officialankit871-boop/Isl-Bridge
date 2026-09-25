"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

from backend.core.config import get_settings
from backend.core.constants import SERVICE_NAME

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response returned by the service health check."""

    status: str
    service: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return the current service status and version."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=settings.app_version,
    )

