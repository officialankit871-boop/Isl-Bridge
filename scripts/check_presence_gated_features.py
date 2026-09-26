"""Verify presence-gated features against V2 inputs and split manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
LANDMARK_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks_v2"
METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "landmark_metadata_v2"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features_presence_gated"
MANIFEST_PATH = FEATURE_DIR / "manifest.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "presence_gated_feature_report.json"
SOURCE_DIMENSION = 225
FEATURE_DIMENSION = 228
SPLITS = ("train", "val", "test")
PRESENCE_KEYS = (
    "left_hand_present",
    "right_hand_present",
    "pose_present",
)
PRESENCE_COLUMNS = {
    "left_hand_presence": 225,
    "right_hand_presence": 226,
    "pose_presence": 227,
}
MANIFEST_COLUMNS = {
    "split",
    "video_path",
    "feature_path",
    "num_frames",
    "feature_dim",
}


def read_split(split: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"landmark_path", "video_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def normalize_video_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def source_key(landmark_path: str) -> str:
    return Path(str(landmark_path).strip().replace("\\", "/")).stem


def resolve_output_path(value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if resolved.parent != FEATURE_DIR.resolve() or resolved.suffix.lower() != ".npy":
        raise ValueError(f"Feature path is outside the expected output directory: {value}")
    return resolved


def verify_one(
    row_number: int,
    row: pd.Series,
    landmark_path_by_video: dict[str, str],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "manifest_row": row_number,
        "split": str(row["split"]),
        "video_path": str(row["video_path"]),
        "feature_path": str(row["feature_path"]),
        "valid": False,
        "issues": [],
    }
    try:
        video_key = normalize_video_path(row["video_path"])
        source_landmark_path = landmark_path_by_video[video_key]
        key = source_key(source_landmark_path)
        feature_path = resolve_output_path(str(row["feature_path"]))
        expected_frames = int(row["num_frames"])
        expected_dim = int(row["feature_dim"])
    except (TypeError, ValueError, OSError) as exc:
        record["issues"].append(f"invalid_manifest_entry: {exc}")
        return record

    if feature_path.stem != key:
        record["issues"].append(
            f"filename_key_mismatch: expected {key}, found {feature_path.stem}"
        )
    if expected_dim != FEATURE_DIMENSION:
        record["issues"].append(f"manifest_feature_dim_invalid: {expected_dim}")
    if not feature_path.is_file():
        record["issues"].append("feature_file_missing")
        return record
    try:
        features = np.load(feature_path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        record["issues"].append(f"feature_load_error: {exc}")
        return record

    record["shape"] = list(features.shape)
    record["dtype"] = str(features.dtype)
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1] != FEATURE_DIMENSION
    ):
        record["issues"].append(
            f"feature_shape_invalid: expected (T, {FEATURE_DIMENSION}), found {features.shape}"
        )
        record["valid"] = False
        return record
    if features.dtype != np.float32:
        record["issues"].append(f"feature_dtype_invalid: {features.dtype}")
    if features.shape[0] != expected_frames:
        record["issues"].append(
            f"frame_count_mismatch: manifest={expected_frames}, file={features.shape[0]}"
        )
    if not np.isfinite(features).all():
        record["issues"].append("feature_contains_nan_or_inf")
        record["valid"] = False
        return record

    presence_flags = features[:, SOURCE_DIMENSION:]
    if not np.isin(presence_flags, (0.0, 1.0)).all():
        record["issues"].append("presence_flags_not_binary")

    landmark_path = LANDMARK_DIR / f"{key}.npy"
    metadata_path = METADATA_DIR / f"{key}.npz"
    try:
        original = np.load(landmark_path, allow_pickle=False)
        if (
            original.shape != (features.shape[0], SOURCE_DIMENSION)
            or original.dtype != np.float32
            or not np.isfinite(original).all()
        ):
            raise ValueError(
                f"invalid source landmark array shape={original.shape}, dtype={original.dtype}"
            )
        with np.load(metadata_path, allow_pickle=False) as metadata:
            missing = [key_name for key_name in PRESENCE_KEYS if key_name not in metadata.files]
            if missing:
                raise ValueError(f"source metadata missing keys: {missing}")
            presence = {key_name: metadata[key_name] for key_name in PRESENCE_KEYS}
        for key_name, values in presence.items():
            if values.dtype != np.bool_ or values.shape != (features.shape[0],):
                raise ValueError(
                    f"invalid source {key_name} shape={values.shape}, dtype={values.dtype}"
                )

        left_present = presence["left_hand_present"]
        right_present = presence["right_hand_present"]
        pose_present = presence["pose_present"]
        if not np.array_equal(presence_flags[:, 0], left_present.astype(np.float32)):
            record["issues"].append("left_hand_presence_flag_mismatch")
        if not np.array_equal(presence_flags[:, 1], right_present.astype(np.float32)):
            record["issues"].append("right_hand_presence_flag_mismatch")
        if not np.array_equal(presence_flags[:, 2], pose_present.astype(np.float32)):
            record["issues"].append("pose_presence_flag_mismatch")

        absent_left = ~left_present
        absent_right = ~right_present
        if np.any(features[absent_left, 99:162] != 0.0):
            record["issues"].append("absent_left_hand_coordinates_not_zero")
        if np.any(features[absent_right, 162:225] != 0.0):
            record["issues"].append("absent_right_hand_coordinates_not_zero")
        if not np.array_equal(features[:, :99], original[:, :99]):
            record["issues"].append("pose_coordinates_changed")
        if np.any(left_present) and not np.array_equal(
            features[left_present, 99:162], original[left_present, 99:162]
        ):
            record["issues"].append("present_left_hand_coordinates_changed")
        if np.any(right_present) and not np.array_equal(
            features[right_present, 162:225], original[right_present, 162:225]
        ):
            record["issues"].append("present_right_hand_coordinates_changed")

        percentages = {
            "left_hand_presence_percent": float(np.mean(left_present) * 100.0),
            "right_hand_presence_percent": float(np.mean(right_present) * 100.0),
            "both_hands_presence_percent": float(
                np.mean(left_present & right_present) * 100.0
            ),
            "pose_presence_percent": float(np.mean(pose_present) * 100.0),
        }
        record["presence_percentages"] = percentages
    except (OSError, ValueError, KeyError, EOFError) as exc:
        record["issues"].append(f"source_verification_error: {exc}")

    record["valid"] = not record["issues"]
    return record


def main() -> int:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Feature manifest not found: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    missing_columns = MANIFEST_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"{MANIFEST_PATH} is missing columns: {sorted(missing_columns)}")

    split_frames = {split: read_split(split) for split in SPLITS}
    expected_counts = {split: len(frame) for split, frame in split_frames.items()}
    actual_counts = {
        split: int((manifest["split"] == split).sum())
        for split in SPLITS
    }
    count_mismatches = {
        split: {"expected": expected_counts[split], "manifest": actual_counts[split]}
        for split in SPLITS
        if expected_counts[split] != actual_counts[split]
    }
    membership_mismatches: dict[str, dict[str, list[str]]] = {}
    for split in SPLITS:
        expected = {
            normalize_video_path(value)
            for value in split_frames[split]["video_path"]
        }
        actual = {
            normalize_video_path(value)
            for value in manifest.loc[manifest["split"] == split, "video_path"]
        }
        if expected != actual:
            membership_mismatches[split] = {
                "missing_videos": sorted(expected - actual),
                "unexpected_videos": sorted(actual - expected),
            }

    landmark_path_by_video: dict[str, str] = {}
    for split_frame in split_frames.values():
        for _, split_row in split_frame.iterrows():
            video_key = normalize_video_path(split_row["video_path"])
            if video_key in landmark_path_by_video:
                raise ValueError(f"Duplicate video_path across split files: {video_key}")
            landmark_path_by_video[video_key] = str(split_row["landmark_path"])

    records = [
        verify_one(index + 2, row, landmark_path_by_video)
        for index, (_, row) in enumerate(manifest.iterrows())
    ]
    duplicate_paths = manifest["feature_path"].duplicated(keep=False)
    for index in manifest.index[duplicate_paths]:
        records[int(index)]["issues"].append("duplicate_manifest_feature_path")
        records[int(index)]["valid"] = False
    unknown_splits = [
        int(index)
        for index, split in manifest["split"].items()
        if str(split) not in SPLITS
    ]
    for index in unknown_splits:
        records[index]["issues"].append("unknown_split")
        records[index]["valid"] = False

    manifested_paths = set()
    for value in manifest["feature_path"]:
        try:
            manifested_paths.add(resolve_output_path(str(value)))
        except ValueError:
            continue
    extra_files = sorted(
        path.name for path in FEATURE_DIR.glob("*.npy") if path.resolve() not in manifested_paths
    )
    invalid_records = [record for record in records if not record["valid"]]
    invalid_count = (
        len(invalid_records)
        + len(extra_files)
        + len(count_mismatches)
        + len(membership_mismatches)
    )

    valid_records = [
        record for record in records
        if record["valid"] and "presence_percentages" in record
    ]
    total_frames = sum(int(record["shape"][0]) for record in valid_records)
    statistics = {
        name: (
            sum(
                record["presence_percentages"][name] * int(record["shape"][0])
                for record in valid_records
            )
            / total_frames
            if total_frames
            else None
        )
        for name in (
            "left_hand_presence_percent",
            "right_hand_presence_percent",
            "both_hands_presence_percent",
            "pose_presence_percent",
        )
    }
    report = {
        "expected_split_counts": expected_counts,
        "manifest_split_counts": actual_counts,
        "total_manifest_features": int(len(manifest)),
        "invalid_files": invalid_count,
        "count_mismatches": count_mismatches,
        "split_membership_mismatches": membership_mismatches,
        "unmanifested_feature_files": extra_files,
        "feature_dimension": FEATURE_DIMENSION,
        "required_dtype": "float32",
        "total_valid_frames": total_frames,
        "frame_weighted_presence_percentages": statistics,
        "files": records,
    }

    print("PRESENCE-GATED FEATURE CHECK")
    print(f"Train: {actual_counts['train']} / expected {expected_counts['train']}")
    print(f"Validation: {actual_counts['val']} / expected {expected_counts['val']}")
    print(f"Test: {actual_counts['test']} / expected {expected_counts['test']}")
    print(f"Total: {len(manifest)}")
    print(f"Invalid files: {invalid_count}")
    print(f"Feature dimension: {FEATURE_DIMENSION}")
    print("Required dtype: float32")
    for name, value in statistics.items():
        label = name.replace("_presence_percent", "").replace("_", " ").title()
        print(f"{label} presence: {value:.2f}%" if value is not None else f"{label} presence: unavailable")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 1 if invalid_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
