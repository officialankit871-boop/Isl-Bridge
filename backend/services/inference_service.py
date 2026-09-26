"""Model-loading and inference service for the CNN+BiLSTM ISL classifier."""

from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.core.config import get_settings
from backend.services.landmark_service import (
    DEFAULT_MAX_FRAMES,
    FEATURE_DIMENSION,
    LandmarkValidationError,
    prepare_sequence_for_inference,
    validate_landmark_sequence,
)
from scripts.cnn_lstm_model import CNNBiLSTMClassifier

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"


class ManifestMappingError(RuntimeError):
    """Raised when a predicted model class has no valid manifest mapping."""


def load_manifest_class_mappings(
    manifest_path: str | Path = MANIFEST_PATH,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load class-name-to-sentence and model-label-to-class-name mappings."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Class sentence manifest not found: {path}")

    class_name_to_sentence: dict[str, str] = {}
    label_id_to_class_name: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required_columns = {"class_name", "label_id", "sentence"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Manifest {path} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}."
            )

        for row_number, row in enumerate(reader, start=2):
            class_name = (row.get("class_name") or "").strip()
            label_id = (row.get("label_id") or "").strip()
            sentence = (row.get("sentence") or "").strip()
            if not class_name or not label_id or not sentence:
                raise ValueError(
                    f"Manifest {path} has an empty class_name, label_id, or sentence "
                    f"at row {row_number}."
                )

            existing_sentence = class_name_to_sentence.setdefault(class_name, sentence)
            existing_class_name = label_id_to_class_name.setdefault(label_id, class_name)
            if existing_sentence != sentence or existing_class_name != class_name:
                raise ValueError(
                    f"Manifest {path} contains conflicting class mappings "
                    f"at row {row_number}."
                )

    if not class_name_to_sentence:
        raise ValueError(f"Manifest {path} contains no class mappings.")
    return class_name_to_sentence, label_id_to_class_name


class ISLInferenceService:
    """Load and reuse the CNN+BiLSTM model for prediction requests."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        settings = get_settings()
        resolved_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else Path(settings.resolved_model_checkpoint_path)
        )
        self.checkpoint_path = resolved_path.resolve(strict=False)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}. "
                "Ensure the model exists and set MODEL_CHECKPOINT_PATH if needed."
            )

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint payload must be a dictionary.")
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint is missing model_state_dict.")
        if "class_to_index" not in checkpoint:
            raise KeyError("Checkpoint is missing class_to_index.")
        if "model_config" not in checkpoint:
            raise KeyError("Checkpoint is missing model_config.")

        self.class_to_index: dict[str, int] = checkpoint["class_to_index"]
        self.index_to_class = self._reverse_mapping(self.class_to_index)
        (
            self.class_name_to_sentence,
            self.label_id_to_class_name,
        ) = load_manifest_class_mappings()
        self.model_config: dict[str, Any] = checkpoint["model_config"]
        self.num_classes = len(self.class_to_index)
        self.input_dim = FEATURE_DIMENSION
        self.max_frames = int(self.model_config.get("max_frames", DEFAULT_MAX_FRAMES))

        if self.num_classes <= 0:
            raise ValueError("Checkpoint must define at least one class.")
        if self.num_classes != 101:
            raise ValueError(f"Expected 101 classes, found {self.num_classes}.")
        if self.input_dim != FEATURE_DIMENSION:
            raise ValueError(
                f"Expected input dimension {FEATURE_DIMENSION}, found {self.input_dim}."
            )

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested, but CUDA is unavailable.")
        self.model = CNNBiLSTMClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            cnn_channels_1=int(self.model_config.get("cnn_channels_1", 64)),
            cnn_channels_2=int(self.model_config.get("cnn_channels_2", 128)),
            lstm_hidden_size=int(self.model_config.get("lstm_hidden_size", 64)),
            lstm_layers=int(self.model_config.get("lstm_layers", 2)),
            cnn_dropout=float(self.model_config.get("cnn_dropout", 0.25)),
            lstm_dropout=float(self.model_config.get("lstm_dropout", 0.30)),
            classifier_dropout=float(self.model_config.get("classifier_dropout", 0.50)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(
            "Loaded CNN-BiLSTM model: classes=%s, input_dim=%s, max_frames=%s, device=%s, checkpoint=%s",
            self.num_classes,
            self.input_dim,
            self.max_frames,
            self.device,
            self.checkpoint_path,
        )

    @staticmethod
    def _reverse_mapping(class_to_index: dict[str, int]) -> dict[int, str]:
        """Return a safe index-to-class mapping."""
        reversed_map: dict[int, str] = {}
        for label, index in class_to_index.items():
            if index in reversed_map:
                raise ValueError(f"Duplicate class index detected: {index}")
            reversed_map[int(index)] = str(label)
        return reversed_map

    def sentence_for_class_id(self, class_id: str) -> str:
        """Resolve a predicted checkpoint label to its manifest sentence."""
        class_name = self.label_id_to_class_name.get(class_id)
        if class_name is None:
            raise ManifestMappingError(
                f"Predicted class '{class_id}' is not present in the manifest "
                f"at {MANIFEST_PATH}."
            )
        sentence = self.class_name_to_sentence.get(class_name)
        if sentence is None:
            raise ManifestMappingError(
                f"Manifest class '{class_name}' for predicted class '{class_id}' "
                "has no sentence."
            )
        return sentence

    def predict(self, landmarks: list[list[float]] | np.ndarray) -> dict[str, Any]:
        """Return a prediction record for a landmark sequence."""
        validated = validate_landmark_sequence(landmarks, max_frames=self.max_frames)
        prepared_features, padding_mask = prepare_sequence_for_inference(
            validated,
            max_frames=self.max_frames,
        )

        features = torch.from_numpy(prepared_features).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(padding_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(features, mask)
        probabilities = torch.softmax(logits[0], dim=0)
        topk_values, topk_indices = torch.topk(probabilities, k=min(5, probabilities.shape[0]))

        top_k: list[dict[str, Any]] = []
        for confidence, index in zip(topk_values.cpu().tolist(), topk_indices.cpu().tolist()):
            label = self.index_to_class.get(int(index), str(index))
            top_k.append(
                {
                    "label": label,
                    "class_id": label,
                    "confidence": float(confidence),
                }
            )

        best_index = int(topk_indices[0].item())
        best_label = self.index_to_class.get(best_index, str(best_index))
        best_confidence = float(topk_values[0].item())
        return {
            "label": best_label,
            "class_id": best_label,
            "confidence": best_confidence,
            "top_k": top_k,
        }


@lru_cache(maxsize=1)
def get_inference_service() -> ISLInferenceService:
    """Return a cached inference service, loading the model only once."""
    return ISLInferenceService()
