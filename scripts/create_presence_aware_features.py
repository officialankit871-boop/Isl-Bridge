"""Append V2 per-frame detector-presence flags to normalized landmarks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
LANDMARK_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks_v2"
METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "landmark_metadata_v2"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "features_presence_aware"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
LOGGER = logging.getLogger("isl_bridge.presence_aware_features")

SPLITS = ("train", "val", "test")
LANDMARK_DIMENSION = 225
FEATURE_DIMENSION = 228
PRESENCE_KEYS = (
    "left_hand_present",
    "right_hand_present",
    "pose_present",
)
MANIFEST_COLUMNS = (
    "split",
    "video_path",
    "feature_path",
    "num_frames",
    "feature_dim",
)


def split_rows(split: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"landmark_path", "video_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["landmark_path"].duplicated().any():
        raise ValueError(f"{path} contains duplicate landmark_path values.")
    return frame


def source_stem(landmark_path: str) -> str:
    normalized = landmark_path.strip().replace("\\", "/")
    stem = Path(normalized).stem
    if not stem or stem in {".", ".."}:
        raise ValueError(f"Invalid landmark_path key: {landmark_path!r}")
    return stem


def load_inputs(stem: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    landmark_path = LANDMARK_DIR / f"{stem}.npy"
    metadata_path = METADATA_DIR / f"{stem}.npz"
    if not landmark_path.is_file():
        raise FileNotFoundError(f"V2 landmark file not found: {landmark_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"V2 metadata file not found: {metadata_path}")

    landmarks = np.load(landmark_path, allow_pickle=False)
    if landmarks.ndim != 2 or landmarks.shape[0] == 0 or landmarks.shape[1] != LANDMARK_DIMENSION:
        raise ValueError(
            f"Expected landmark shape (T, {LANDMARK_DIMENSION}) in "
            f"{landmark_path}; found {landmarks.shape}."
        )
    if landmarks.dtype != np.float32:
        raise ValueError(
            f"Expected float32 landmarks in {landmark_path}; found {landmarks.dtype}."
        )
    if not np.isfinite(landmarks).all():
        raise ValueError(f"Landmarks contain NaN/Inf: {landmark_path}")

    metadata_arrays: dict[str, np.ndarray] = {}
    with np.load(metadata_path, allow_pickle=False) as metadata:
        missing = [key for key in PRESENCE_KEYS if key not in metadata.files]
        if missing:
            raise ValueError(f"{metadata_path} is missing arrays: {missing}")
        for key in PRESENCE_KEYS:
            values = metadata[key]
            if values.dtype != np.bool_:
                raise ValueError(
                    f"{key} in {metadata_path} must be boolean; found {values.dtype}."
                )
            if values.shape != (landmarks.shape[0],):
                raise ValueError(
                    f"{key} in {metadata_path} must have shape "
                    f"({landmarks.shape[0]},); found {values.shape}."
                )
            metadata_arrays[key] = values
    return landmarks, metadata_arrays


def create_features(
    landmarks: np.ndarray,
    presence: dict[str, np.ndarray],
) -> np.ndarray:
    presence_features = np.column_stack(
        [presence[key].astype(np.float32, copy=False) for key in PRESENCE_KEYS]
    )
    features = np.concatenate((landmarks, presence_features), axis=1)
    if features.shape != (landmarks.shape[0], FEATURE_DIMENSION):
        raise ValueError(f"Unexpected output shape: {features.shape}")
    if features.dtype != np.float32:
        raise ValueError(f"Unexpected output dtype: {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("Combined features contain NaN/Inf.")
    return features


def write_feature(path: Path, features: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing feature file: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        np.save(file, features, allow_pickle=False)


def process_split(split: str, rows: pd.DataFrame) -> tuple[list[dict[str, Any]], int]:
    manifest_rows: list[dict[str, Any]] = []
    errors = 0
    for row_index, row in rows.iterrows():
        video_path = str(row["video_path"]).strip()
        try:
            stem = source_stem(str(row["landmark_path"]))
            features = create_features(*load_inputs(stem))
            output_path = OUTPUT_DIR / f"{stem}.npy"
            write_feature(output_path, features)
            feature_relative_path = output_path.relative_to(PROJECT_ROOT).as_posix()
            manifest_rows.append(
                {
                    "split": split,
                    "video_path": video_path,
                    "feature_path": feature_relative_path,
                    "num_frames": int(features.shape[0]),
                    "feature_dim": FEATURE_DIMENSION,
                }
            )
        except (OSError, ValueError, KeyError, EOFError) as exc:
            errors += 1
            LOGGER.error(
                "Failed split=%s row=%s video=%s: %s",
                split,
                row_index + 2,
                video_path,
                exc,
            )
    return manifest_rows, errors


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if MANIFEST_PATH.exists():
        LOGGER.error(
            "Refusing to overwrite existing manifest: %s. "
            "Move to a new output location before rerunning.",
            MANIFEST_PATH,
        )
        return 2

    try:
        splits = {split: split_rows(split) for split in SPLITS}
    except (FileNotFoundError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2

    all_stems: dict[str, str] = {}
    for split, rows in splits.items():
        for value in rows["landmark_path"]:
            try:
                stem = source_stem(str(value))
            except ValueError as exc:
                LOGGER.error("Invalid key in %s split: %s", split, exc)
                return 2
            previous_split = all_stems.get(stem)
            if previous_split is not None:
                LOGGER.error(
                    "Duplicate feature filename key %s appears in %s and %s.",
                    stem,
                    previous_split,
                    split,
                )
                return 2
            all_stems[stem] = split

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_manifest_rows: list[dict[str, Any]] = []
    total_errors = 0
    for split in SPLITS:
        LOGGER.info("Processing %s split (%d videos).", split, len(splits[split]))
        rows, errors = process_split(split, splits[split])
        all_manifest_rows.extend(rows)
        total_errors += errors
        LOGGER.info(
            "%s split complete: %d created, %d errors.",
            split,
            len(rows),
            errors,
        )

    manifest = pd.DataFrame(all_manifest_rows, columns=MANIFEST_COLUMNS)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Features created: {len(all_manifest_rows)}")
    print(f"Errors: {total_errors}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
