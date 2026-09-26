"""Focused tests for the real-time inference pipeline."""

from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from backend.services.inference_service import ISLInferenceService
from backend.services.landmark_service import LandmarkValidationError
from scripts.realtime_inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HISTORY_SIZE,
    DEFAULT_MAX_FRAMES,
    FEATURE_DIMENSION,
    BodyNormalizer,
    append_temporal_frame,
    build_feature_vector,
    load_checkpoint,
    prepare_model_input,
    run_prediction,
    smooth_prediction,
)


class TestRealtimeInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service, cls.device = load_checkpoint(DEFAULT_CHECKPOINT, force_cpu=True)

    def test_checkpoint_load_and_synthetic_inference(self) -> None:
        service: ISLInferenceService = self.service
        device: torch.device = self.device
        self.assertEqual(service.input_dim, 225)
        self.assertEqual(service.num_classes, 101)
        sequence = np.random.default_rng(42).normal(
            size=(DEFAULT_MAX_FRAMES, FEATURE_DIMENSION)
        ).astype(np.float32)
        predictions = run_prediction(sequence, service, device, DEFAULT_MAX_FRAMES)
        self.assertGreaterEqual(len(predictions), 1)
        self.assertLessEqual(len(predictions), 5)
        self.assertIn(predictions[0]["class_id"], service.class_to_index)
        self.assertTrue(predictions[0]["sentence"])
        self.assertGreaterEqual(predictions[0]["confidence"], 0.0)
        self.assertLessEqual(predictions[0]["confidence"], 1.0)

    def test_backend_padding_masks_for_supported_lengths(self) -> None:
        for valid_frames in (32, 64, 128):
            with self.subTest(valid_frames=valid_frames):
                sequence = np.ones((valid_frames, FEATURE_DIMENSION), dtype=np.float32)
                features, padding_mask = prepare_model_input(
                    sequence, DEFAULT_MAX_FRAMES
                )
                self.assertEqual(
                    features.shape, (DEFAULT_MAX_FRAMES, FEATURE_DIMENSION)
                )
                self.assertEqual(int((~padding_mask).sum()), valid_frames)
                self.assertFalse(padding_mask[:valid_frames].any())
                self.assertTrue(padding_mask[valid_frames:].all())
                self.assertTrue(np.all(features[valid_frames:] == 0.0))

    def test_manifest_sentence_mapping(self) -> None:
        service: ISLInferenceService = self.service
        class_name = service.label_id_to_class_name["class_019"]
        self.assertEqual(
            service.class_name_to_sentence[class_name],
            "He is going into the room",
        )

    def test_prediction_smoothing_requires_majority(self) -> None:
        history: deque[tuple[str, float]] = deque(maxlen=DEFAULT_HISTORY_SIZE)
        history.extend(
            [("class_001", 0.6), ("class_002", 0.7), ("class_001", 0.8)]
        )
        self.assertIsNone(smooth_prediction(history, DEFAULT_HISTORY_SIZE))
        history.append(("class_001", 0.9))
        stable = smooth_prediction(history, DEFAULT_HISTORY_SIZE)
        self.assertIsNotNone(stable)
        self.assertEqual(stable[0], "class_001")
        self.assertAlmostEqual(stable[1], 0.7666666667)

    def test_nonfinite_model_input_is_rejected(self) -> None:
        for invalid_value in (np.nan, np.inf):
            with self.subTest(invalid_value=invalid_value):
                sequence = np.zeros((32, FEATURE_DIMENSION), dtype=np.float32)
                sequence[0, 0] = invalid_value
                with self.assertRaisesRegex(LandmarkValidationError, "finite"):
                    prepare_model_input(sequence, DEFAULT_MAX_FRAMES)

    def test_temporal_buffer_is_bounded(self) -> None:
        buffer: deque[np.ndarray] = deque(maxlen=DEFAULT_MAX_FRAMES)
        for frame_id in range(DEFAULT_MAX_FRAMES + 7):
            append_temporal_frame(
                buffer,
                np.full(FEATURE_DIMENSION, frame_id, dtype=np.float32),
                DEFAULT_MAX_FRAMES,
            )
        self.assertEqual(len(buffer), DEFAULT_MAX_FRAMES)
        self.assertTrue(np.all(buffer[0] == 7.0))

    def test_empty_mediapipe_result_creates_finite_features(self) -> None:
        result = SimpleNamespace(
            pose_landmarks=None,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
        )
        features, hands_detected = build_feature_vector(result, BodyNormalizer())
        self.assertEqual(features.shape, (FEATURE_DIMENSION,))
        self.assertTrue(np.isfinite(features).all())
        self.assertFalse(hands_detected)


if __name__ == "__main__":
    unittest.main()
