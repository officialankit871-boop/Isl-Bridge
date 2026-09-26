"""Utilities for validating and preparing landmark sequences for inference."""

from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_DIMENSION = 225
DEFAULT_MAX_FRAMES = 128


class LandmarkValidationError(ValueError):
    """Raised when sequence payloads do not match the expected ISL contract."""


def validate_landmark_sequence(
    sequence: Any,
    *,
    expected_dim: int = FEATURE_DIMENSION,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_sequence_length: int = 4096,
) -> np.ndarray:
    """Validate a raw landmark sequence and normalize it to float32."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise LandmarkValidationError(
            f"Landmark sequence must be a 2D array with shape (T, {expected_dim})."
        )
    if array.shape[0] == 0:
        raise LandmarkValidationError("Landmark sequence cannot be empty.")
    if array.shape[1] != expected_dim:
        raise LandmarkValidationError(
            f"Each frame must contain exactly {expected_dim} values; got {array.shape[1]}."
        )
    if array.shape[0] > max_sequence_length:
        raise LandmarkValidationError(
            f"Sequence is too long: {array.shape[0]} frames exceeds the maximum of {max_sequence_length}."
        )
    if not np.isfinite(array).all():
        raise LandmarkValidationError("Landmark values must be finite numbers only.")
    if array.shape[0] > max_frames:
        return array[:max_frames]
    return array


def prepare_sequence_for_inference(
    sequence: np.ndarray,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare a valid sequence to the exact model input shape: (128, 225)."""
    validated = validate_landmark_sequence(sequence, max_frames=max_frames)
    original_length = validated.shape[0]
    if original_length > max_frames:
        working = validated[:max_frames].copy()
        valid_length = max_frames
    else:
        working = np.zeros((max_frames, FEATURE_DIMENSION), dtype=np.float32)
        working[:original_length] = validated
        valid_length = original_length

    padding_mask = np.ones(max_frames, dtype=bool)
    padding_mask[:valid_length] = False
    return working.astype(np.float32, copy=False), padding_mask
