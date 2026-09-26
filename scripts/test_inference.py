"""Direct inference smoke test for the current CNN+BiLSTM checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.inference_service import ISLInferenceService
from backend.services.landmark_service import prepare_sequence_for_inference


def main() -> int:
    """Load a real landmark file and print the top-5 predictions."""
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "cnn_lstm_best.pt"
    landmark_path = PROJECT_ROOT / "data" / "processed" / "landmarks" / "000000.npy"

    try:
        service = ISLInferenceService(checkpoint_path)
        sequence = np.load(landmark_path, allow_pickle=False)
        if sequence.ndim != 2:
            raise ValueError(f"Expected 2D landmark array, got {sequence.shape}.")
        if sequence.shape[1] != 225:
            raise ValueError(f"Expected 225 features per frame, got {sequence.shape[1]}.")
        if sequence.shape[0] == 0:
            raise ValueError("Landmark sequence is empty.")
        if not np.isfinite(sequence).all():
            raise ValueError("Landmark sequence contains NaN or Inf values.")
        prepared, padding_mask = prepare_sequence_for_inference(sequence, max_frames=service.max_frames)
        prediction = service.predict(sequence)
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as exc:
        raise SystemExit(f"Inference validation failed: {exc}") from exc

    print(f"Device: {service.device}")
    print(f"Checkpoint path: {service.checkpoint_path}")
    print(f"Number of classes: {service.num_classes}")
    print(f"Input shape: {tuple(sequence.shape)}")
    print(f"Prepared shape: {tuple(prepared.shape)}")
    print(f"Predicted class: {prediction['label']}")
    print(f"Class index: {prediction['class_id']}")
    print(f"Confidence: {prediction['confidence']:.6f}")
    print("Top-5 predictions:")
    for item in prediction["top_k"]:
        print(f"  - {item['label']} | {item['class_id']} | {item['confidence']:.6f}")
    print(f"Padding mask valid entries: {int((~padding_mask).sum())} / {padding_mask.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
