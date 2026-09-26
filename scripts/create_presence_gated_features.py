"""Create V2 landmark features gated by per-frame hand detector presence."""

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
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "features_presence_gated"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
LOGGER = logging.getLogger("isl_bridge.presence_gated_features")

SPLITS = ("train", "val", "test")
SOURCE_DIMENSION = 225
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


def read_split(split: str) -> pd.DataFrame:
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


def filename_key(landmark_path: str) -> str:
    normalized = str(landmark_path).strip().replace("\\", "/")
    key = Path(normalized).stem
    if not key or key in {".", ".."}:
        raise ValueError(f"Invalid landmark filename key: {landmark_path!r}")
    return key


def load_sources(key: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    landmark_path = LANDMARK_DIR / f"{key}.npy"
    metadata_path = METADATA_DIR / f"{key}.npz"
    if not landmark_path.is_file():
        raise FileNotFoundError(f"V2 landmark file not found: {landmark_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"V2 metadata file not found: {metadata_path}")

    landmarks = np.load(landmark_path, allow_pickle=False)
    if (
        landmarks.ndim != 2
        or landmarks.shape[0] == 0
        or landmarks.shape[1] != SOURCE_DIMENSION
    ):
        raise ValueError(
            f"Expected non-empty (T, {SOURCE_DIMENSION}) in {landmark_path}; "
            f"found {landmarks.shape}."
        )
    if landmarks.dtype != np.float32:
        raise ValueError(
            f"Expected float32 landmarks in {landmark_path}; found {landmarks.dtype}."
        )
    if not np.isfinite(landmarks).all():
        raise ValueError(f"Landmarks contain NaN/Inf: {landmark_path}")

    presence: dict[str, np.ndarray] = {}
    with np.load(metadata_path, allow_pickle=False) as metadata:
        missing = [name for name in PRESENCE_KEYS if name not in metadata.files]
        if missing:
            raise ValueError(f"{metadata_path} is missing arrays: {missing}")
        for name in PRESENCE_KEYS:
            values = metadata[name]
            if values.dtype != np.bool_:
                raise ValueError(
                    f"{name} in {metadata_path} must be boolean; found {values.dtype}."
                )
            if values.shape != (landmarks.shape[0],):
                raise ValueError(
                    f"{name} in {metadata_path} must have shape "
                    f"({landmarks.shape[0]},); found {values.shape}."
                )
            presence[name] = values
    return landmarks, presence


def gate_features(
    landmarks: np.ndarray,
    presence: dict[str, np.ndarray],
) -> np.ndarray:
    gated = landmarks.copy()
    gated[~presence["left_hand_present"], 99:162] = 0.0
    gated[~presence["right_hand_present"], 162:225] = 0.0
    presence_features = np.column_stack(
        [presence[key].astype(np.float32, copy=False) for key in PRESENCE_KEYS]
    )
    features = np.concatenate((gated, presence_features), axis=1)
    if features.shape != (landmarks.shape[0], FEATURE_DIMENSION):
        raise ValueError(f"Unexpected output shape: {features.shape}")
    if features.dtype != np.float32:
        raise ValueError(f"Unexpected output dtype: {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("Output contains NaN/Inf.")
    return features


def write_new_feature(path: Path, features: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing feature file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        np.save(file, features, allow_pickle=False)


def process_split(split: str, frame: pd.DataFrame) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    errors = 0
    for row_index, row in frame.iterrows():
        video_path = str(row["video_path"]).strip()
        try:
            key = filename_key(str(row["landmark_path"]))
            landmarks, presence = load_sources(key)
            features = gate_features(landmarks, presence)
            output_path = OUTPUT_DIR / f"{key}.npy"
            write_new_feature(output_path, features)
            rows.append(
                {
                    "split": split,
                    "video_path": video_path,
                    "feature_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
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
    return rows, errors


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if MANIFEST_PATH.exists():
        LOGGER.error("Refusing to overwrite existing manifest: %s", MANIFEST_PATH)
        return 2
    try:
        split_frames = {split: read_split(split) for split in SPLITS}
    except (FileNotFoundError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2

    keys_to_splits: dict[str, str] = {}
    for split, frame in split_frames.items():
        for raw_path in frame["landmark_path"]:
            try:
                key = filename_key(str(raw_path))
            except ValueError as exc:
                LOGGER.error("Invalid path key in %s split: %s", split, exc)
                return 2
            if key in keys_to_splits:
                LOGGER.error(
                    "Feature filename key %s is duplicated across %s and %s.",
                    key,
                    keys_to_splits[key],
                    split,
                )
                return 2
            keys_to_splits[key] = split

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    errors = 0
    for split in SPLITS:
        LOGGER.info("Processing %s split (%d videos).", split, len(split_frames[split]))
        rows, split_errors = process_split(split, split_frames[split])
        manifest_rows.extend(rows)
        errors += split_errors
        LOGGER.info(
            "%s complete: %d created, %d errors.",
            split,
            len(rows),
            split_errors,
        )

    if MANIFEST_PATH.exists():
        LOGGER.error("Manifest appeared during extraction; refusing to overwrite it.")
        return 2
    pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS).to_csv(
        MANIFEST_PATH, index=False
    )
    print(f"Features created: {len(manifest_rows)}")
    print(f"Errors: {errors}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
