"""Extract normalized pose and hand landmarks from sentence-level videos."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("isl_bridge.landmarks")
LANDMARK_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
PROCESSING_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_manifest.csv"
FEATURE_DIMENSION = 225
LANDMARK_GROUPS = (
    ("pose", 33),
    ("left_hand", 21),
    ("right_hand", 21),
)
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12


@dataclass
class ExtractionResult:
    """Result and quality counters for one manifest row."""

    landmark_path: str
    video_path: str
    relative_video_path: str
    file_name: str
    sentence: str | None
    class_name: str | None
    label_id: str | None
    gloss: str | None
    signer: str | None
    num_frames: int
    feature_dim: int
    processing_status: str
    valid_frame_count: int
    frames_with_pose: int
    frames_with_left_hand: int
    frames_with_right_hand: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse extraction command-line options."""
    parser = argparse.ArgumentParser(
        description="Extract normalized MediaPipe Holistic pose and hand landmarks."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sorted manifest rows.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess existing landmark files instead of reusing valid files.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure extraction logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_manifest(limit: int | None) -> pd.DataFrame:
    """Load and deterministically sort the M2 valid-video manifest."""
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"video_path", "relative_video_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    manifest = manifest.sort_values("relative_video_path", kind="stable").reset_index(drop=True)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer.")
        manifest = manifest.head(limit).copy()
    return manifest


def import_mediapipe() -> Any:
    """Import MediaPipe or raise an actionable environment error."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed in the active environment. "
            "Install a Python-3.13-compatible MediaPipe release explicitly, "
            "then rerun: python -m pip install \"mediapipe>=0.10.20,<0.11\""
        ) from exc
    return mp


def empty_landmark_group(count: int) -> np.ndarray:
    """Return zero-filled coordinates for a missing landmark group."""
    return np.zeros((count, 3), dtype=np.float32)


def extract_group(result: Any, count: int) -> tuple[np.ndarray, bool]:
    """Convert a MediaPipe landmark list to coordinates and a presence flag."""
    if result is None or not getattr(result, "landmark", None):
        return empty_landmark_group(count), False
    coordinates = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in result.landmark],
        dtype=np.float32,
    )
    if coordinates.shape != (count, 3):
        return empty_landmark_group(count), False
    return coordinates, True


class BodyNormalizer:
    """Normalize coordinates around the shoulder midpoint and shoulder scale.

    For each frame, the shoulder center is subtracted from all coordinates and
    the result is divided by the Euclidean distance between the shoulders.
    When either shoulder is unavailable, the previous valid center and scale
    are reused; before any valid reference exists, center zero and scale one
    are used. This prevents NaN and infinity values without interpolation.
    """

    def __init__(self) -> None:
        self.previous_center = np.zeros(3, dtype=np.float32)
        self.previous_scale = 1.0
        self.has_reference = False

    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """Return a finite float32 normalized landmark frame."""
        left = landmarks[POSE_LEFT_SHOULDER]
        right = landmarks[POSE_RIGHT_SHOULDER]
        if np.any(left != 0.0) and np.any(right != 0.0):
            center = ((left + right) / 2.0).astype(np.float32)
            scale = float(np.linalg.norm(left - right))
            if np.isfinite(scale) and scale > 1e-6:
                self.previous_center = center
                self.previous_scale = scale
                self.has_reference = True
        center = self.previous_center if self.has_reference else np.zeros(3, dtype=np.float32)
        scale = self.previous_scale if self.has_reference else 1.0
        normalized = (landmarks - center) / scale
        return np.nan_to_num(
            normalized.astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


def extract_video(video_path: Path, mp: Any) -> tuple[np.ndarray, dict[str, int]]:
    """Process one video sequentially and return its normalized landmark array."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("video_unreadable")
    frames: list[np.ndarray] = []
    counters = {
        "valid_frame_count": 0,
        "frames_with_pose": 0,
        "frames_with_left_hand": 0,
        "frames_with_right_hand": 0,
    }
    normalizer = BodyNormalizer()
    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:
            while True:
                read_ok, frame = capture.read()
                if not read_ok:
                    break
                counters["valid_frame_count"] += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb_frame)
                pose, has_pose = extract_group(result.pose_landmarks, 33)
                left, has_left = extract_group(result.left_hand_landmarks, 21)
                right, has_right = extract_group(result.right_hand_landmarks, 21)
                counters["frames_with_pose"] += int(has_pose)
                counters["frames_with_left_hand"] += int(has_left)
                counters["frames_with_right_hand"] += int(has_right)
                combined = np.concatenate((pose, left, right), axis=0)
                frames.append(normalizer.normalize(combined).reshape(FEATURE_DIMENSION))
    finally:
        capture.release()
    if not frames:
        raise ValueError("video_has_no_readable_frames")
    return np.asarray(frames, dtype=np.float32), counters


def validate_array(array: np.ndarray) -> None:
    """Validate the persisted landmark array contract."""
    if array.dtype != np.float32:
        raise ValueError(f"unexpected_dtype:{array.dtype}")
    if array.ndim != 2 or array.shape[1] != FEATURE_DIMENSION or array.shape[0] <= 0:
        raise ValueError(f"unexpected_shape:{array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("non_finite_values")


def relative_output_path(index: int) -> str:
    """Return the deterministic relative filename for a manifest row."""
    return f"{index:06d}.npy"


def base_result(row: pd.Series, landmark_path: str) -> dict[str, Any]:
    """Copy manifest metadata into a processing result."""
    return {
        "landmark_path": landmark_path,
        "video_path": str(row.get("video_path", "")),
        "relative_video_path": str(row.get("relative_video_path", "")),
        "file_name": str(row.get("file_name", "")),
        "sentence": str(row.get("sentence", "")) or None,
        "class_name": str(row.get("class_name", "")) or None,
        "label_id": str(row.get("label_id", "")) or None,
        "gloss": str(row.get("gloss", "")) or None,
        "signer": str(row.get("signer", "")) or None,
    }


def process_row(index: int, row: pd.Series, force: bool, mp: Any) -> ExtractionResult:
    """Process or reuse one manifest row and return its processing metadata."""
    output_name = relative_output_path(index)
    output_path = LANDMARK_OUTPUT_DIR / output_name
    result = base_result(row, f"data/processed/landmarks/{output_name}")
    if output_path.is_file() and not force:
        try:
            existing = np.load(output_path, allow_pickle=False)
            validate_array(existing)
            return ExtractionResult(
                **result,
                num_frames=int(existing.shape[0]),
                feature_dim=int(existing.shape[1]),
                processing_status="skipped",
                valid_frame_count=int(existing.shape[0]),
                frames_with_pose=0,
                frames_with_left_hand=0,
                frames_with_right_hand=0,
            )
        except (OSError, ValueError) as exc:
            LOGGER.warning("Existing output %s is invalid; reprocessing: %s", output_path, exc)
    video_path = Path(str(row["video_path"]))
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    try:
        array, counters = extract_video(video_path, mp)
        validate_array(array)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, array, allow_pickle=False)
        return ExtractionResult(
            **result,
            num_frames=int(array.shape[0]),
            feature_dim=int(array.shape[1]),
            processing_status="processed",
            **counters,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.exception("Failed to process %s", video_path)
        return ExtractionResult(
            **result,
            num_frames=0,
            feature_dim=FEATURE_DIMENSION,
            processing_status="failed",
            valid_frame_count=0,
            frames_with_pose=0,
            frames_with_left_hand=0,
            frames_with_right_hand=0,
            error=str(exc),
        )


def write_processing_manifest(results: list[ExtractionResult]) -> None:
    """Write the deterministic landmark processing manifest."""
    PROCESSING_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.__dict__ for result in results]).to_csv(
        PROCESSING_MANIFEST_PATH, index=False
    )


def print_report(results: list[ExtractionResult]) -> None:
    """Print aggregate extraction results without printing landmark data."""
    successful = [result for result in results if result.processing_status in {"processed", "skipped"}]
    processed = sum(result.processing_status == "processed" for result in results)
    skipped = sum(result.processing_status == "skipped" for result in results)
    failed = sum(result.processing_status == "failed" for result in results)
    total_frames = sum(result.num_frames for result in successful)
    denominator = total_frames or 1
    print("\n" + "=" * 40)
    print("LANDMARK EXTRACTION REPORT")
    print("=" * 40)
    print(f"Total videos: {len(results)}")
    print(f"Processed successfully: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Total frames: {total_frames}")
    print(f"Average frames/video: {total_frames / len(successful) if successful else 0:.2f}")
    print(f"Pose detection rate: {sum(r.frames_with_pose for r in successful) / denominator:.4f}")
    print(f"Left hand detection rate: {sum(r.frames_with_left_hand for r in successful) / denominator:.4f}")
    print(f"Right hand detection rate: {sum(r.frames_with_right_hand for r in successful) / denominator:.4f}")
    print(f"Output directory: {LANDMARK_OUTPUT_DIR}")
    print(f"Manifest: {PROCESSING_MANIFEST_PATH}")
    print("=" * 40)


def main() -> int:
    """Run deterministic landmark extraction for the requested manifest subset."""
    configure_logging()
    arguments = parse_args()
    try:
        manifest = load_manifest(arguments.limit)
        mp = import_mediapipe()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    LANDMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[ExtractionResult] = []
    for index, (_, row) in enumerate(manifest.iterrows()):
        print(f"Processing {index + 1}/{len(manifest)}", flush=True)
        results.append(process_row(index, row, arguments.force, mp))
    write_processing_manifest(results)
    print_report(results)
    return 0 if all(result.processing_status != "failed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
