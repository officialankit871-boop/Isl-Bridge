"""Prediction endpoint for landmark-sequence inference."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, field_validator

from backend.services.inference_service import (
    ManifestMappingError,
    get_inference_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])


class PredictionItem(BaseModel):
    """Single predicted class record."""

    label: str
    class_id: str
    confidence: float
    sentence: str


class PredictRequest(BaseModel):
    """Request payload containing a landmark sequence."""

    model_config = ConfigDict(extra="forbid")

    landmarks: list[list[float]]

    @field_validator("landmarks")
    @classmethod
    def validate_landmarks(cls, value: list[list[float]]) -> list[list[float]]:
        """Validate the landmark payload shape and content."""
        if not value:
            raise ValueError("landmarks must not be empty.")
        if len(value) > 4096:
            raise ValueError("landmarks sequence is too long.")
        for frame in value:
            if not isinstance(frame, list):
                raise ValueError("Each landmark frame must be a list of floats.")
            if len(frame) != 225:
                raise ValueError("Each frame must contain exactly 225 values.")
            if not all(isinstance(value_, (int, float)) for value_ in frame):
                raise ValueError("Each landmark feature must be numeric.")
            finite = np.asarray(frame, dtype=np.float32)
            if not np.isfinite(finite).all():
                raise ValueError("Landmark values must be finite numbers only.")
        return value


class PredictResponse(BaseModel):
    """Inference response payload returned by the API."""

    success: bool
    prediction: PredictionItem
    top_k: list[PredictionItem]
    metadata: dict[str, Any]


@router.post("/predict", response_model=PredictResponse)
async def predict_landmarks(payload: PredictRequest, request: Request) -> PredictResponse:
    """Run inference for a landmark sequence using the cached CNN+BiLSTM model."""
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        service = get_inference_service()

    try:
        prediction = service.predict(payload.landmarks)
    except ValueError as exc:
        logger.warning("Rejecting invalid prediction payload: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive error handling
        logger.exception("Prediction failed for request.")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    try:
        prediction_sentence = service.sentence_for_class_id(prediction["class_id"])
        top_k = [
            PredictionItem(
                label=item["label"],
                class_id=item["class_id"],
                confidence=float(item["confidence"]),
                sentence=service.sentence_for_class_id(item["class_id"]),
            )
            for item in prediction["top_k"]
        ]
    except ManifestMappingError as exc:
        logger.exception("Prediction class could not be mapped to a manifest sentence.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(
        success=True,
        prediction=PredictionItem(
            label=prediction["label"],
            class_id=prediction["class_id"],
            confidence=float(prediction["confidence"]),
            sentence=prediction_sentence,
        ),
        top_k=top_k,
        metadata={
            "frames_received": len(payload.landmarks),
            "frames_used": service.max_frames,
            "feature_dimension": service.input_dim,
            "model": "cnn_lstm",
        },
    )
