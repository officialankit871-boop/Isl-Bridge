"""Exact sentence-level sign video retrieval endpoint."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.core.config import get_settings
from backend.services.sign_retriever import SignRetriever
from backend.services.text_mapper import TextMapper

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
    video_id: str
    video_url: str


class SignTranslateResponse(BaseModel):
    matched: bool
    match_type: str
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


@lru_cache(maxsize=1)
def get_text_mapper() -> TextMapper:
    """Create one process-wide mapper using the existing cached retriever."""
    return TextMapper(get_sign_retriever())


def video_id_for_path(video_path: str) -> str:
    """Create a stable opaque identifier from an indexed video path."""
    return hashlib.sha256(video_path.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def get_indexed_video_map() -> dict[str, str]:
    """Map trusted stable IDs to CSV relative paths using the cached index."""
    retriever = get_sign_retriever()
    return {
        video_id_for_path(video["video_path"]): video["relative_video_path"]
        for video in retriever.iter_videos()
    }


@router.post(
    "/sign/translate",
    response_model=SignTranslateResponse,
    summary="Map supported text to an Indian Sign Language video mapping.",
    description=(
        "Retrieve an exact sentence-level Indian Sign Language video mapping, "
        "or resolve a reviewed CSV alias to one. The endpoint does not perform fuzzy translation."
    ),
)
async def translate_sign(payload: SignTranslateRequest, request: Request) -> dict[str, Any]:
    """Return the sentence metadata and every indexed video variant."""
    retriever = getattr(request.app.state, "sign_retriever", None)
    if retriever is None:
        retriever = get_sign_retriever()
    mapper = getattr(request.app.state, "text_mapper", None) or get_text_mapper()
    mapping = mapper.map_text(payload.text)
    if mapping["matched"]:
        result = retriever.find_sentence(mapping["canonical_sentence"])
    else:
        result = retriever.find_sentence(payload.text)
    result["match_type"] = mapping["match_type"]
    for video in result["videos"]:
        video_id = video_id_for_path(video["video_path"])
        video["video_id"] = video_id
        video["video_url"] = f"/sign/video/{video_id}"
    return result


@router.get("/sign/video/{video_id}", response_class=FileResponse, include_in_schema=False)
async def get_sign_video(video_id: str) -> FileResponse:
    """Stream only a video row found in the trusted sign index."""
    relative_path = get_indexed_video_map().get(video_id)
    if relative_path is None:
        raise HTTPException(status_code=404, detail="Video not found.")

    dataset_root = get_settings().resolved_dataset_root.resolve()
    video_path = (dataset_root / relative_path).resolve()
    try:
        video_path.relative_to(dataset_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Video not found.") from exc
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)


__all__ = ["router", "get_sign_retriever", "get_text_mapper", "video_id_for_path"]
