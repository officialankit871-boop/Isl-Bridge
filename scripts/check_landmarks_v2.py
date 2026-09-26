"""Validate V2 landmark and per-frame detector metadata outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANDMARK_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks_v2"
METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "landmark_metadata_v2"
REPORT_PATH = PROJECT_ROOT / "reports" / "landmark_v2_check.json"
FEATURE_DIMENSION = 225
PRESENCE_KEYS = (
    "left_hand_present",
    "right_hand_present",
    "pose_present",
)


def percentage(values: np.ndarray) -> float:
    return float(np.mean(values) * 100.0)


def inspect_pair(landmark_path: Path, metadata_path: Path) -> dict[str, Any]:
    issues: list[str] = []
    try:
        landmarks = np.load(landmark_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        return {"filename": landmark_path.name, "valid": False, "issues": [f"landmark_load_error: {exc}"]}

    if (
        landmarks.ndim != 2
        or landmarks.shape[0] == 0
        or landmarks.shape[1] != FEATURE_DIMENSION
    ):
        issues.append(f"invalid_landmark_shape: {list(landmarks.shape)}")
        frame_count = int(landmarks.shape[0]) if landmarks.ndim else 0
    else:
        frame_count = int(landmarks.shape[0])
    if landmarks.dtype != np.float32:
        issues.append(f"invalid_landmark_dtype: {landmarks.dtype}")
    if not np.issubdtype(landmarks.dtype, np.number):
        issues.append("landmarks_are_not_numeric; NaN/Inf check unavailable")
    elif not np.isfinite(landmarks).all():
        issues.append("landmarks_contain_nan_or_inf")

    try:
        with np.load(metadata_path, allow_pickle=False) as metadata:
            absent = [key for key in PRESENCE_KEYS if key not in metadata.files]
            if absent:
                issues.append(f"missing_presence_arrays: {absent}")
            arrays: dict[str, np.ndarray] = {}
            for key in PRESENCE_KEYS:
                if key not in metadata.files:
                    continue
                values = metadata[key]
                arrays[key] = values
                if values.dtype != np.bool_:
                    issues.append(f"{key}_is_not_boolean: {values.dtype}")
                if values.shape != (frame_count,):
                    issues.append(
                        f"{key}_length_mismatch: expected {(frame_count,)}, found {values.shape}"
                    )
            optional_visibility = [
                key for key in metadata.files if key.endswith("_visibility")
            ]
            for key in optional_visibility:
                values = metadata[key]
                if values.ndim != 2 or values.shape[0] != frame_count:
                    issues.append(f"{key}_length_mismatch: found {values.shape}")

    except (OSError, ValueError) as exc:
        return {
            "filename": landmark_path.name,
            "valid": False,
            "frames": frame_count,
            "issues": issues + [f"metadata_load_error: {exc}"],
        }

    percentages: dict[str, float] = {}
    if all(key in arrays for key in PRESENCE_KEYS):
        expected_shapes = all(arrays[key].shape == (frame_count,) for key in PRESENCE_KEYS)
        boolean_dtypes = all(arrays[key].dtype == np.bool_ for key in PRESENCE_KEYS)
        if expected_shapes and boolean_dtypes and frame_count:
            percentages = {
                "left_hand_presence_percent": percentage(arrays["left_hand_present"]),
                "right_hand_presence_percent": percentage(arrays["right_hand_present"]),
                "both_hands_presence_percent": percentage(
                    arrays["left_hand_present"] & arrays["right_hand_present"]
                ),
                "pose_presence_percent": percentage(arrays["pose_present"]),
            }
    return {
        "filename": landmark_path.name,
        "valid": not issues,
        "frames": frame_count,
        "shape": list(landmarks.shape),
        "dtype": str(landmarks.dtype),
        "issues": issues,
        **percentages,
    }


def main() -> int:
    landmark_paths = sorted(LANDMARK_DIR.glob("*.npy")) if LANDMARK_DIR.is_dir() else []
    metadata_paths = sorted(METADATA_DIR.glob("*.npz")) if METADATA_DIR.is_dir() else []
    landmarks_by_stem = {path.stem: path for path in landmark_paths}
    metadata_by_stem = {path.stem: path for path in metadata_paths}
    matching_stems = sorted(set(landmarks_by_stem) & set(metadata_by_stem))
    landmark_only = sorted(set(landmarks_by_stem) - set(metadata_by_stem))
    metadata_only = sorted(set(metadata_by_stem) - set(landmarks_by_stem))

    records = [
        inspect_pair(landmarks_by_stem[stem], metadata_by_stem[stem])
        for stem in matching_stems
    ]
    invalid_files = (
        len(landmark_only)
        + len(metadata_only)
        + sum(not record["valid"] for record in records)
    )
    metric_keys = (
        "left_hand_presence_percent",
        "right_hand_presence_percent",
        "both_hands_presence_percent",
        "pose_presence_percent",
    )
    valid_records = [record for record in records if record["valid"]]
    total_valid_frames = sum(record["frames"] for record in valid_records)
    means = {
        key: (
            sum(
                record[key] * record["frames"] / 100.0
                for record in valid_records
            )
            / total_valid_frames
            * 100.0
            if total_valid_frames
            else None
        )
        for key in metric_keys
    }
    report = {
        "videos_processed": len(matching_stems),
        "landmark_files": len(landmark_paths),
        "metadata_files": len(metadata_paths),
        "matching_filenames": len(matching_stems),
        "landmark_without_metadata": landmark_only,
        "metadata_without_landmark": metadata_only,
        "invalid_files": invalid_files,
        "mean_presence_percentages": means,
        "files": records,
    }

    print("V2 EXTRACTION CHECK")
    print(f"Videos processed: {len(matching_stems)}")
    print(f"Landmark files: {len(landmark_paths)}")
    print(f"Metadata files: {len(metadata_paths)}")
    print(f"Matching filenames: {len(matching_stems)}")
    print(f"Invalid files: {invalid_files}")
    print(
        "Mean left-hand presence: "
        f"{means['left_hand_presence_percent']:.2f}%"
        if means["left_hand_presence_percent"] is not None
        else "Mean left-hand presence: unavailable"
    )
    print(
        "Mean right-hand presence: "
        f"{means['right_hand_presence_percent']:.2f}%"
        if means["right_hand_presence_percent"] is not None
        else "Mean right-hand presence: unavailable"
    )
    print(
        "Mean both-hand presence: "
        f"{means['both_hands_presence_percent']:.2f}%"
        if means["both_hands_presence_percent"] is not None
        else "Mean both-hand presence: unavailable"
    )
    print(
        "Mean pose presence: "
        f"{means['pose_presence_percent']:.2f}%"
        if means["pose_presence_percent"] is not None
        else "Mean pose presence: unavailable"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 1 if invalid_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
