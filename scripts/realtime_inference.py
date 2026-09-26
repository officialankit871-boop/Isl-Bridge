"""Run real-time ISL sign-to-text inference from a webcam."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.inference_service import ISLInferenceService  # noqa: E402
from backend.services.landmark_service import (  # noqa: E402
    FEATURE_DIMENSION,
    prepare_sequence_for_inference,
)
from scripts.extract_landmarks import (  # noqa: E402
    BodyNormalizer,
    extract_group,
    import_mediapipe,
)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "cnn_lstm_best.pt"
DEFAULT_MAX_FRAMES = 128
DEFAULT_INFERENCE_INTERVAL = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.15
DEFAULT_HISTORY_SIZE = 5
NUM_CLASSES = 101


@dataclass
class DisplayPrediction:
    """Latest stable result and optional raw top-k predictions."""

    sentence: str = "Waiting for stable prediction"
    confidence: float | None = None
    status: str = "Warming up"
    top_k: list[dict[str, Any]] = field(default_factory=list)


class RollingFPS:
    """Calculate a rolling event rate over a short time window."""

    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_seconds = window_seconds
        self.timestamps: deque[float] = deque()

    def tick(self, now: float | None = None) -> float:
        """Record an event and return its current rolling rate."""
        current = time.perf_counter() if now is None else now
        self.timestamps.append(current)
        while (
            len(self.timestamps) > 1
            and current - self.timestamps[0] > self.window_seconds
        ):
            self.timestamps.popleft()
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if len(self.timestamps) < 2 or elapsed <= 0.0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


def parse_args() -> argparse.Namespace:
    """Parse real-time inference options."""
    parser = argparse.ArgumentParser(
        description="Run real-time ISL sign-to-text inference."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="CNN-BiLSTM checkpoint path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Temporal window length; must match the trained checkpoint.",
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=DEFAULT_INFERENCE_INTERVAL,
        help="Run model inference every N frames after warm-up.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Display low-confidence results as Uncertain.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=DEFAULT_HISTORY_SIZE,
        help="Number of recent predictions used for majority smoothing.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force model inference to run on CPU.",
    )
    parser.add_argument(
        "--show-top5",
        action="store_true",
        help="Show the five highest-probability classes at startup.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, trained_max_frames: int) -> None:
    """Reject CLI settings that would violate the trained input contract."""
    if args.camera < 0:
        raise ValueError("--camera must be a non-negative device index.")
    if args.max_frames < 1:
        raise ValueError("--max-frames must be positive.")
    if args.max_frames != trained_max_frames:
        raise ValueError(
            f"--max-frames must match the checkpoint value "
            f"({trained_max_frames}); received {args.max_frames}."
        )
    if args.inference_interval < 1:
        raise ValueError("--inference-interval must be positive.")
    if args.history_size < 1:
        raise ValueError("--history-size must be positive.")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0.0 and 1.0.")


def load_checkpoint(
    checkpoint_path: str | Path,
    force_cpu: bool = False,
) -> tuple[ISLInferenceService, torch.device]:
    """Load the existing inference service and verify its baseline dimensions."""
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    requested_device = torch.device("cpu") if force_cpu else None
    service = ISLInferenceService(path, device=requested_device)
    input_shape = tuple(service.model.input_norm.normalized_shape)
    output_classes = service.model.classifier.out_features
    if input_shape != (FEATURE_DIMENSION,):
        raise ValueError(
            f"Checkpoint input dimension must be {FEATURE_DIMENSION}; "
            f"model expects {input_shape}."
        )
    if service.num_classes != NUM_CLASSES or output_classes != NUM_CLASSES:
        raise ValueError(
            f"Checkpoint must define exactly {NUM_CLASSES} classes; "
            f"found {service.num_classes}."
        )
    if len(service.index_to_class) != NUM_CLASSES or set(
        service.index_to_class
    ) != set(range(NUM_CLASSES)):
        raise ValueError("Checkpoint class_to_index must map indices 0 through 100.")

    trained_max_frames = int(service.model_config.get("max_frames", DEFAULT_MAX_FRAMES))
    if trained_max_frames < 1:
        raise ValueError("Checkpoint model_config has an invalid max_frames value.")
    for class_id in service.class_to_index:
        service.sentence_for_class_id(class_id)

    service.model.eval()
    return service, service.device


def build_feature_vector(
    result: Any,
    normalizer: BodyNormalizer,
) -> tuple[np.ndarray, bool]:
    """Extract and normalize one frame with the offline extraction pipeline."""
    pose, has_pose = extract_group(result.pose_landmarks, 33)
    left, has_left = extract_group(result.left_hand_landmarks, 21)
    right, has_right = extract_group(result.right_hand_landmarks, 21)
    combined = np.concatenate((pose, left, right), axis=0)
    features = normalizer.normalize(combined).reshape(FEATURE_DIMENSION)
    if features.shape != (FEATURE_DIMENSION,):
        raise ValueError(
            f"Expected {FEATURE_DIMENSION} features from MediaPipe; "
            f"received {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError("MediaPipe produced NaN or Inf landmark features.")
    return features.astype(np.float32, copy=False), has_left or has_right


def append_temporal_frame(
    buffer: deque[np.ndarray],
    features: np.ndarray,
    max_frames: int,
) -> None:
    """Append a validated feature vector to a bounded rolling buffer."""
    if features.shape != (FEATURE_DIMENSION,):
        raise ValueError(
            f"Expected a ({FEATURE_DIMENSION},) frame; got {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Temporal buffer cannot accept NaN or Inf features.")
    if buffer.maxlen != max_frames:
        raise ValueError("Temporal buffer maxlen does not match max_frames.")
    buffer.append(features)


def prepare_model_input(
    sequence: np.ndarray,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the backend's deterministic truncation and zero-padding behavior."""
    prepared, padding_mask = prepare_sequence_for_inference(
        sequence,
        max_frames=max_frames,
    )
    if prepared.shape != (max_frames, FEATURE_DIMENSION):
        raise ValueError(
            f"Prepared input must have shape ({max_frames}, {FEATURE_DIMENSION}); "
            f"got {prepared.shape}."
        )
    if not np.isfinite(prepared).all():
        raise ValueError("Prepared model input contains NaN or Inf.")
    return prepared, padding_mask


def run_prediction(
    sequence: np.ndarray,
    service: ISLInferenceService,
    device: torch.device,
    max_frames: int,
) -> list[dict[str, Any]]:
    """Run inference and resolve top-k class names through the manifest."""
    prepared, padding_mask = prepare_model_input(sequence, max_frames)
    features_tensor = torch.from_numpy(prepared).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(padding_mask).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = service.model(features_tensor, mask_tensor)
        probabilities = torch.softmax(logits[0], dim=0)
        values, indices = torch.topk(
            probabilities,
            k=min(5, service.num_classes),
        )

    results: list[dict[str, Any]] = []
    for confidence, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        class_id = service.index_to_class[int(index)]
        results.append(
            {
                "class_id": class_id,
                "sentence": service.sentence_for_class_id(class_id),
                "confidence": float(confidence),
            }
        )
    return results


def smooth_prediction(
    history: deque[tuple[str, float]],
    history_size: int,
) -> tuple[str, float] | None:
    """Return a class only after it wins a strict majority of the history."""
    if not history:
        return None
    counts = Counter(class_id for class_id, _ in history)
    most_recent_position = {
        class_id: max(
            index
            for index, (candidate, _) in enumerate(history)
            if candidate == class_id
        )
        for class_id in counts
    }
    winner = max(
        counts,
        key=lambda class_id: (
            counts[class_id],
            most_recent_position[class_id],
        ),
    )
    required_votes = history_size // 2 + 1
    if counts[winner] < required_votes:
        return None
    confidence = float(
        np.mean(
            [
                score
                for class_id, score in history
                if class_id == winner
            ]
        )
    )
    return winner, confidence


def wrap_text(text: str, max_characters: int) -> list[str]:
    """Wrap overlay text at word boundaries for typical webcam widths."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + len(word) + 1 > max_characters:
            lines.append(current)
            current = word
        else:
            current += f" {word}"
    lines.append(current)
    return lines


def draw_overlay(
    frame: np.ndarray,
    prediction: DisplayPrediction,
    *,
    camera_fps: float,
    inference_fps: float,
    device: torch.device,
    buffer_size: int,
    max_frames: int,
    hands_detected: bool,
    show_top5: bool,
    confidence_threshold: float,
) -> np.ndarray:
    """Draw recognition status and optional top-five results on a copy."""
    output = frame.copy()
    panel_width = min(output.shape[1] - 16, 620)
    line_height = 26
    sentence_lines = wrap_text(prediction.sentence, max(20, panel_width // 12))
    top_lines: list[str] = []
    if show_top5:
        top_lines = [
            f"{index}. {item['sentence']} — {item['confidence'] * 100:.1f}%"
            for index, item in enumerate(prediction.top_k[:5], start=1)
        ]
    content_lines = (
        11
        + len(sentence_lines)
        + (1 + len(top_lines) if show_top5 and top_lines else 0)
    )
    panel_height = min(output.shape[0] - 16, content_lines * line_height + 20)
    if panel_width <= 0 or panel_height <= 0:
        return output

    overlay = output.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (15, 20, 28), -1)
    cv2.addWeighted(overlay, 0.78, output, 0.22, 0, output)

    x = 20
    y = 34
    lines = [
        ("ISL BRIDGE — REAL-TIME SIGN RECOGNITION", (80, 220, 255)),
        (f"Prediction: {sentence_lines[0]}", (245, 245, 245)),
    ]
    lines.extend((line, (245, 245, 245)) for line in sentence_lines[1:])
    confidence_label = (
        f"{prediction.confidence * 100:.1f}%"
        if prediction.confidence is not None
        else "--"
    )
    lines.extend(
        [
            (f"Confidence: {confidence_label}", (180, 245, 180)),
            (f"Threshold: {confidence_threshold:.2f}", (220, 220, 220)),
            (f"Camera FPS: {camera_fps:.1f}", (220, 220, 220)),
            (
                f"Inference FPS: {inference_fps:.1f}" if inference_fps else "Inference FPS: --",
                (220, 220, 220),
            ),
            (f"Device: {device}", (220, 220, 220)),
            (f"Buffer: {buffer_size} / {max_frames}", (220, 220, 220)),
            (f"Status: {prediction.status}", (80, 220, 255)),
            (f"Hands: {'Detected' if hands_detected else 'Not detected'}", (220, 220, 220)),
            ("Q / ESC Quit | R Reset | S Top-5 | C Clear", (200, 200, 200)),
        ]
    )
    if show_top5 and top_lines:
        lines.append(("Top Predictions:", (80, 220, 255)))
        lines.extend((line, (245, 245, 245)) for line in top_lines)

    max_visible_lines = max(1, (panel_height - 12) // line_height)
    for line_index, (text, color) in enumerate(lines[:max_visible_lines]):
        cv2.putText(
            output,
            text,
            (x, y + line_index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 if line_index == 0 else 0.54,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def print_startup(
    service: ISLInferenceService,
    device: torch.device,
    camera_index: int,
    max_frames: int,
    inference_interval: int,
    confidence_threshold: float,
) -> None:
    """Print the active inference configuration and keyboard controls."""
    print("-----------------------------------------")
    print("ISL BRIDGE")
    print("Real-Time Sign Recognition")
    print("-----------------------------------------")
    print(f"Checkpoint: {service.checkpoint_path}")
    print(f"Classes: {service.num_classes}")
    print(f"Input dimension: {service.input_dim}")
    print(f"Max frames: {max_frames}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Camera: {camera_index}")
    print(f"Inference interval: every {inference_interval} frames")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    print("Controls: Q / ESC Quit | R Reset | S Toggle Top-5 | C Clear")
    print("Starting webcam...")


def run_webcam(
    service: ISLInferenceService,
    device: torch.device,
    *,
    camera_index: int,
    max_frames: int,
    inference_interval: int,
    confidence_threshold: float,
    history_size: int,
    show_top5: bool,
) -> int:
    """Capture, normalize, buffer, classify, and display webcam frames."""
    mp = import_mediapipe()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        print(
            "ERROR: Could not open webcam.\n"
            "Possible reasons:\n"
            "- camera already in use\n"
            "- incorrect camera index\n"
            "- Windows camera permission\n"
            "- another application using the webcam",
            file=sys.stderr,
        )
        return 1

    buffer: deque[np.ndarray] = deque(maxlen=max_frames)
    prediction_history: deque[tuple[str, float]] = deque(maxlen=history_size)
    display = DisplayPrediction()
    camera_rate = RollingFPS()
    inference_rate = RollingFPS()
    camera_fps = 0.0
    inference_fps = 0.0
    frames_since_reset = 0
    hands_detected = False
    window_name = "ISL Bridge — Real-Time Sign Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:
            normalizer = BodyNormalizer()
            while True:
                read_ok, frame = capture.read()
                if not read_ok:
                    print("ERROR: Webcam frame read failed.", file=sys.stderr)
                    break
                camera_fps = camera_rate.tick()
                frames_since_reset += 1

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb_frame)
                features, hands_detected = build_feature_vector(result, normalizer)
                append_temporal_frame(buffer, features, max_frames)

                warmed_up = len(buffer) == max_frames
                inference_due = (
                    warmed_up
                    and (frames_since_reset - max_frames) % inference_interval == 0
                )
                if inference_due:
                    top_k = run_prediction(
                        np.asarray(buffer, dtype=np.float32),
                        service,
                        device,
                        max_frames,
                    )
                    inference_fps = inference_rate.tick()
                    display.top_k = top_k
                    best = top_k[0]
                    if best["confidence"] < confidence_threshold:
                        prediction_history.clear()
                        display.sentence = "Uncertain"
                        display.confidence = float(best["confidence"])
                        display.status = "Uncertain (below threshold)"
                    else:
                        prediction_history.append(
                            (str(best["class_id"]), float(best["confidence"]))
                        )
                        stable = smooth_prediction(prediction_history, history_size)
                        if stable is None:
                            display.confidence = float(best["confidence"])
                            display.status = "Stabilizing prediction"
                            if not display.sentence or display.sentence == "Uncertain":
                                display.sentence = "Stabilizing..."
                        else:
                            class_id, stable_confidence = stable
                            display.sentence = service.sentence_for_class_id(class_id)
                            display.confidence = stable_confidence
                            display.status = "Detecting"
                elif not warmed_up:
                    display.status = "Warming up"

                rendered = draw_overlay(
                    frame,
                    display,
                    camera_fps=camera_fps,
                    inference_fps=inference_fps,
                    device=device,
                    buffer_size=len(buffer),
                    max_frames=max_frames,
                    hands_detected=hands_detected,
                    show_top5=show_top5,
                    confidence_threshold=confidence_threshold,
                )
                cv2.imshow(window_name, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    show_top5 = not show_top5
                elif key == ord("c"):
                    prediction_history.clear()
                    display = DisplayPrediction(
                        sentence="Waiting for stable prediction",
                        status="Prediction cleared",
                    )
                elif key == ord("r"):
                    buffer.clear()
                    prediction_history.clear()
                    frames_since_reset = 0
                    normalizer = BodyNormalizer()
                    display = DisplayPrediction()
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    """Load the baseline model and start the webcam recognition loop."""
    args = parse_args()
    service, device = load_checkpoint(args.checkpoint, force_cpu=args.cpu)
    trained_max_frames = int(service.model_config.get("max_frames", DEFAULT_MAX_FRAMES))
    validate_args(args, trained_max_frames)
    print_startup(
        service,
        device,
        args.camera,
        args.max_frames,
        args.inference_interval,
        args.confidence_threshold,
    )
    return run_webcam(
        service,
        device,
        camera_index=args.camera,
        max_frames=args.max_frames,
        inference_interval=args.inference_interval,
        confidence_threshold=args.confidence_threshold,
        history_size=args.history_size,
        show_top5=args.show_top5,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Closing real-time inference.")
        raise SystemExit(0)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
