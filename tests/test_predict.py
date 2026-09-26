"""Integration tests for the prediction endpoint."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.inference_service import load_manifest_class_mappings

client = TestClient(app)


def _load_landmarks() -> list[list[float]]:
    """Load the packaged landmark sample used for live inference testing."""
    array = np.load("data/processed/landmarks/000000.npy", allow_pickle=False).astype(np.float32)
    if array.ndim != 2 or array.shape[1] != 225:
        raise AssertionError(f"Unexpected landmark shape: {array.shape}")
    if not np.isfinite(array).all():
        raise AssertionError("Landmark array contains invalid values.")
    return array.tolist()


def test_predict_returns_prediction_for_real_landmark_sequence() -> None:
    """POST /predict should infer from the real model checkpoint."""
    response = client.post("/predict", json={"landmarks": _load_landmarks()})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True
    assert "prediction" in payload
    assert "top_k" in payload
    assert "metadata" in payload
    assert payload["prediction"]["label"]
    assert payload["prediction"]["class_id"]
    assert payload["prediction"]["sentence"]
    class_name_to_sentence, label_id_to_class_name = load_manifest_class_mappings()
    predicted_class_name = label_id_to_class_name[payload["prediction"]["class_id"]]
    assert payload["prediction"]["sentence"] == class_name_to_sentence[predicted_class_name]
    confidence = float(payload["prediction"]["confidence"])
    assert 0.0 <= confidence <= 1.0
    assert len(payload["top_k"]) <= 5
    for item in payload["top_k"]:
        class_name = label_id_to_class_name[item["class_id"]]
        assert item["sentence"] == class_name_to_sentence[class_name]
    assert payload["metadata"]["feature_dimension"] == 225
    assert payload["metadata"]["model"] == "cnn_lstm"


def test_manifest_maps_class_019_to_sentence() -> None:
    """The manifest entry for class_019 provides its human-readable sentence."""
    class_name_to_sentence, label_id_to_class_name = load_manifest_class_mappings()
    class_name = label_id_to_class_name["class_019"]
    assert class_name_to_sentence[class_name] == "He is going into the room"


def test_predict_rejects_invalid_payload() -> None:
    """Malformed payloads should be rejected with a client-safe error."""
    response = client.post("/predict", json={"landmarks": [[1.0, 2.0]]})
    assert response.status_code in {400, 422}
