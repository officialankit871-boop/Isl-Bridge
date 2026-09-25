"""Dataset and temporal batching utilities for landmark sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIMENSION = 225


def resolve_path(value: str) -> Path:
    """Resolve a project-relative landmark path."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def temporal_prepare(
    sequence: np.ndarray,
    max_frames: int,
    training: bool,
    rng: np.random.Generator | None = None,
    landmark_noise_std: float = 0.0,
    frame_dropout_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop, augment training sequences, and zero-pad with a padding mask."""
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"Expected (T, {FEATURE_DIMENSION}), got {sequence.shape}")
    sequence = sequence.astype(np.float32, copy=False)
    length = sequence.shape[0]
    if length > max_frames:
        start = int(rng.integers(0, length - max_frames + 1)) if training and rng else 0
        sequence = sequence[start:start + max_frames]
    valid_length = min(sequence.shape[0], max_frames)
    output = np.zeros((max_frames, FEATURE_DIMENSION), dtype=np.float32)
    output[:valid_length] = sequence[:valid_length]
    padding_mask = np.ones(max_frames, dtype=bool)
    padding_mask[:valid_length] = False
    if training and rng is not None and valid_length:
        valid = output[:valid_length]
        # Add small noise only to real landmarks, never to padded frames.
        if landmark_noise_std > 0:
            valid += rng.normal(
                0.0, landmark_noise_std, size=valid.shape
            ).astype(np.float32)
        # Mask complete real frames while preserving the padding mask.
        if frame_dropout_prob > 0:
            dropped = rng.random(valid_length) < frame_dropout_prob
            valid[dropped] = 0.0
    return output, padding_mask


class LandmarkDataset(Dataset[dict[str, torch.Tensor]]):
    """Load variable-length landmark arrays from one split CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        max_frames: int,
        class_to_index: dict[str, int],
        training: bool = False,
        seed: int = 42,
        landmark_noise_std: float = 0.0,
        frame_dropout_prob: float = 0.0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path, keep_default_na=False)
        required = {"landmark_path", "label_id", "feature_dim"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")
        self.max_frames = max_frames
        self.class_to_index = class_to_index
        self.training = training
        self.seed = seed
        self.landmark_noise_std = landmark_noise_std
        self.frame_dropout_prob = frame_dropout_prob

    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load, validate, crop/pad, and return one training example."""
        row = self.frame.iloc[index]
        path = resolve_path(str(row["landmark_path"]))
        try:
            sequence = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not load landmark file {path}: {exc}") from exc
        if sequence.dtype != np.float32:
            sequence = sequence.astype(np.float32)
        if not np.isfinite(sequence).all():
            raise ValueError(f"Non-finite landmark values in {path}")
        if self.training:
            rng = np.random.default_rng()
        else:
            rng = None
        features, padding_mask = temporal_prepare(
            sequence,
            self.max_frames,
            self.training,
            rng,
            self.landmark_noise_std,
            self.frame_dropout_prob,
        )
        label_id = str(row["label_id"])
        if label_id not in self.class_to_index:
            raise ValueError(f"Unknown label_id {label_id} in {self.csv_path}")
        return {
            "features": torch.from_numpy(features),
            "padding_mask": torch.from_numpy(padding_mask),
            "label": torch.tensor(self.class_to_index[label_id], dtype=torch.long),
        }


def class_mapping(frame: pd.DataFrame) -> dict[str, int]:
    """Create a deterministic numeric mapping from label IDs."""
    labels = sorted(frame["label_id"].astype(str).unique())
    return {label: index for index, label in enumerate(labels)}


def training_class_weights(
    frame: pd.DataFrame,
    class_to_index: dict[str, int],
    max_class_weight: float = 3.0,
) -> torch.Tensor:
    """Calculate inverse-frequency class weights using training data only."""
    counts = frame["label_id"].astype(str).value_counts()
    weights = np.ones(len(class_to_index), dtype=np.float32)
    total = len(frame)
    for label, index in class_to_index.items():
        count = int(counts.get(label, 0))
        weights[index] = total / (len(class_to_index) * count) if count else 0.0
    return torch.clamp(
        torch.tensor(weights, dtype=torch.float32), max=max_class_weight
    )
