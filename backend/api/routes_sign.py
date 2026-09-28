"""Exact sentence-level sign video retrieval endpoint."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.services.sign_retriever import SignRetriever

router = APIRouter(tags=["sign"])


class SignTranslateRequest(BaseModel):
    """Sentence text to look up in the trusted sign video index."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="Sentence to retrieve from the ISL dataset.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace only.")
        return value


class SignVideo(BaseModel):
    video_path: str
    relative_video_path: str
    file_name: str


class SignTranslateResponse(BaseModel):
    matched: bool
    sentence: str
    sentence_normalized: str
    gloss: str | None
    class_name: str | None
    label_id: str | None
    video_count: int
    videos: list[SignVideo]


@lru_cache(maxsize=1)
def get_sign_retriever() -> SignRetriever:
    """Create one process-wide retriever on demand; loading errors propagate."""
    return SignRetriever()


@router.post(
    "/sign/translate",
    response_model=SignTranslateResponse,
    summary="Retrieve an exact sentence-level Indian Sign Language video mapping.",
    description=(
        "Retrieve an exact sentence-level Indian Sign Language video mapping. "
        "This endpoint uses exact sentence retrieval and does not perform "
        "semantic or fuzzy translation."
    ),
)
async def translate_sign(payload: SignTranslateRequest, request: Request) -> dict[str, Any]:
    """Return the sentence metadata and every indexed video variant."""
    retriever = getattr(request.app.state, "sign_retriever", None)
    if retriever is None:
        retriever = get_sign_retriever()
    return retriever.find_sentence(payload.text)


__all__ = ["router", "get_sign_retriever"]
