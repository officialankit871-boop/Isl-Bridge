"""Extract normalized landmarks with per-frame detector metadata."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_landmarks import (  # noqa: E402
    FEATURE_DIMENSION,
    BodyNormalizer,
    extract_group,
)

LANDMARK_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks_v2"
METADATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "landmark_metadata_v2"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
LOGGER = logging.getLogger("isl_bridge.landmarks_v2")
LANDMARK_GROUP_SIZES = {
    "pose": 33,
    "left_hand": 21,
    "right_hand": 21,
}
PRESENCE_KEYS = (
    "pose_present",
    "left_hand_present",
    "right_hand_present",
)
VISIBILITY_KEYS = (
    "pose_visibility",
    "left_hand_visibility",
    "right_hand_visibility",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small, split-scoped V2 landmark diagnostic with "
            "per-frame MediaPipe presence metadata."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum videos to extract (default: 5 for a small diagnostic run).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing V2 landmark and metadata outputs.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="Use only rows in this existing split (default: train).",
    )
    return parser.parse_args()


def import_mediapipe() -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed in the active Python environment."
        ) from exc
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "The installed MediaPipe package does not expose mp.solutions, "
            "which the existing extraction pipeline requires."
        )
    return mp


def normalize_key(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def load_selected_rows(split: str, limit: int) -> list[tuple[int, pd.Series]]:
    if limit < 1:
        raise ValueError("--limit must be a positive integer.")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    required = {"video_path", "relative_video_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    manifest = manifest.sort_values(
        "relative_video_path", kind="stable"
    ).reset_index(drop=True)
    manifest_rows = {
        normalize_key(row["relative_video_path"]): (index, row)
        for index, (_, row) in enumerate(manifest.iterrows())
    }
    if len(manifest_rows) != len(manifest):
        raise ValueError("Manifest contains duplicate relative_video_path values.")

    split_path = SPLIT_DIR / f"{split}.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split_frame = pd.read_csv(split_path, keep_default_na=False)
    if "relative_video_path" not in split_frame.columns:
        raise ValueError(f"{split_path} must contain relative_video_path.")

    selected: list[tuple[int, pd.Series]] = []
    missing_rows: list[str] = []
    seen: set[str] = set()
    for value in split_frame["relative_video_path"]:
        key = normalize_key(value)
        if key in seen:
            raise ValueError(f"Duplicate path in {split_path}: {value}")
        seen.add(key)
        item = manifest_rows.get(key)
        if item is None:
            missing_rows.append(str(value))
        else:
            selected.append(item)
    if missing_rows:
        preview = ", ".join(missing_rows[:5])
        raise ValueError(
            f"{len(missing_rows)} rows in {split_path} were not found in the "
            f"manifest; examples: {preview}"
        )
    return selected[:limit]


def visibility_for_group(result: Any, count: int) -> list[float | None] | None:
    """Read real per-landmark visibility values where MediaPipe exposes them."""
    if result is None:
        return None
    landmarks = getattr(result, "landmark", None)
    if not landmarks or len(landmarks) != count:
        return None
    values: list[float | None] = []
    for landmark in landmarks:
        has_field = getattr(landmark, "HasField", None)
        if callable(has_field):
            try:
                if not has_field("visibility"):
                    values.append(None)
                    continue
            except (AttributeError, ValueError):
                values.append(None)
                continue
        value = getattr(landmark, "visibility", None)
        if value is None:
            values.append(None)
            continue
        try:
            visibility = float(value)
        except (TypeError, ValueError):
            values.append(None)
            continue
        values.append(visibility if np.isfinite(visibility) else None)
    return values if any(value is not None for value in values) else None


def optional_visibility_array(
    frame_values: list[list[float | None] | None],
    count: int,
) -> np.ndarray | None:
    if not any(values is not None for values in frame_values):
        return None
    output = np.full((len(frame_values), count), np.nan, dtype=np.float32)
    for frame_index, values in enumerate(frame_values):
        if values is None:
            continue
        output[frame_index] = [
            np.nan if value is None else value for value in values
        ]
    return output


def extract_video_with_metadata(
    video_path: Path,
    mp: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("video_unreadable")

    frames: list[np.ndarray] = []
    presence = {key: [] for key in PRESENCE_KEYS}
    visibility_frames: dict[str, list[list[float | None] | None]] = {
        key: [] for key in VISIBILITY_KEYS
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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb_frame)

                pose_result = result.pose_landmarks
                left_result = result.left_hand_landmarks
                right_result = result.right_hand_landmarks
                pose, has_pose = extract_group(pose_result, LANDMARK_GROUP_SIZES["pose"])
                left, has_left = extract_group(left_result, LANDMARK_GROUP_SIZES["left_hand"])
                right, has_right = extract_group(
                    right_result, LANDMARK_GROUP_SIZES["right_hand"]
                )

                presence["pose_present"].append(has_pose)
                presence["left_hand_present"].append(has_left)
                presence["right_hand_present"].append(has_right)
                for key, component_result, count in (
                    ("pose_visibility", pose_result, LANDMARK_GROUP_SIZES["pose"]),
                    ("left_hand_visibility", left_result, LANDMARK_GROUP_SIZES["left_hand"]),
                    ("right_hand_visibility", right_result, LANDMARK_GROUP_SIZES["right_hand"]),
                ):
                    visibility_frames[key].append(
                        visibility_for_group(component_result, count)
                    )

                combined = np.concatenate((pose, left, right), axis=0)
                frames.append(
                    normalizer.normalize(combined).reshape(FEATURE_DIMENSION)
                )
    finally:
        capture.release()

    if not frames:
        raise ValueError("video_has_no_readable_frames")
    landmarks = np.asarray(frames, dtype=np.float32)
    metadata = {
        key: np.asarray(values, dtype=np.bool_)
        for key, values in presence.items()
    }
    for key, values in visibility_frames.items():
        group_name = key.removesuffix("_visibility")
        visibility = optional_visibility_array(
            values, LANDMARK_GROUP_SIZES[group_name]
        )
        if visibility is not None:
            metadata[key] = visibility
    return landmarks, metadata


def output_paths(manifest_index: int) -> tuple[Path, Path]:
    filename = f"{manifest_index:06d}"
    return (
        LANDMARK_OUTPUT_DIR / f"{filename}.npy",
        METADATA_OUTPUT_DIR / f"{filename}.npz",
    )


def valid_existing_pair(landmark_path: Path, metadata_path: Path) -> bool:
    try:
        landmarks = np.load(landmark_path, allow_pickle=False)
        if (
            landmarks.dtype != np.float32
            or landmarks.ndim != 2
            or landmarks.shape[0] == 0
            or landmarks.shape[1] != FEATURE_DIMENSION
            or not np.isfinite(landmarks).all()
        ):
            return False
        with np.load(metadata_path, allow_pickle=False) as metadata:
            for key in PRESENCE_KEYS:
                values = metadata[key]
                if values.dtype != np.bool_ or values.shape != (landmarks.shape[0],):
                    return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def save_outputs(
    landmark_path: Path,
    metadata_path: Path,
    landmarks: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> None:
    landmark_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with landmark_path.open("wb") as file:
        np.save(file, landmarks, allow_pickle=False)
    with metadata_path.open("wb") as file:
        np.savez_compressed(file, **metadata)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    arguments = parse_args()
    try:
        rows = load_selected_rows(arguments.split, arguments.limit)
        if not rows:
            raise ValueError(f"No videos found in the {arguments.split} split.")
        mp = import_mediapipe()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    processed = 0
    skipped = 0
    failed = 0
    print(
        f"V2 diagnostic extraction: split={arguments.split}, "
        f"limit={arguments.limit}, selected={len(rows)}"
    )
    for number, (manifest_index, row) in enumerate(rows, start=1):
        landmark_path, metadata_path = output_paths(manifest_index)
        if landmark_path.exists() or metadata_path.exists():
            if (
                not arguments.force
                and landmark_path.is_file()
                and metadata_path.is_file()
                and valid_existing_pair(landmark_path, metadata_path)
            ):
                skipped += 1
                print(f"{number}/{len(rows)} SKIPPED {landmark_path.name}")
                continue
            if not arguments.force:
                LOGGER.error(
                    "Incomplete or invalid V2 output for %s; use --force to replace it.",
                    landmark_path.stem,
                )
                failed += 1
                continue

        video_path = Path(str(row["video_path"]))
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        try:
            landmarks, metadata = extract_video_with_metadata(video_path, mp)
            save_outputs(landmark_path, metadata_path, landmarks, metadata)
            processed += 1
            print(
                f"{number}/{len(rows)} PROCESSED {landmark_path.name} "
                f"frames={landmarks.shape[0]}"
            )
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.exception("Failed to process %s", video_path)
            failed += 1
            print(f"{number}/{len(rows)} FAILED {video_path}: {exc}")

    print("\nV2 EXTRACTION SUMMARY")
    print(f"Selected videos: {len(rows)}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Landmarks: {LANDMARK_OUTPUT_DIR}")
    print(f"Metadata: {METADATA_OUTPUT_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
