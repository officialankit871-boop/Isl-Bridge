"""Verify presence-aware feature outputs against their split manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features_presence_aware"
MANIFEST_PATH = FEATURE_DIR / "manifest.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "presence_aware_feature_report.json"
FEATURE_DIMENSION = 228
SPLITS = ("train", "val", "test")
PRESENCE_COLUMNS = (
    ("left_hand_presence", 225),
    ("right_hand_presence", 226),
    ("pose_presence", 227),
)
REQUIRED_MANIFEST_COLUMNS = {
    "split",
    "video_path",
    "feature_path",
    "num_frames",
    "feature_dim",
}


def load_split_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        path = SPLIT_DIR / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Split file not found: {path}")
        counts[split] = len(pd.read_csv(path, keep_default_na=False))
    return counts


def resolve_feature_path(value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    feature_root = FEATURE_DIR.resolve()
    if resolved.parent != feature_root:
        raise ValueError(f"Feature path is outside the output directory: {value}")
    return resolved


def inspect_row(row_number: int, row: pd.Series) -> dict[str, Any]:
    record: dict[str, Any] = {
        "manifest_row": row_number,
        "split": str(row["split"]),
        "feature_path": str(row["feature_path"]),
        "expected_frames": None,
        "valid": False,
        "issues": [],
    }
    try:
        path = resolve_feature_path(str(row["feature_path"]))
    except (OSError, ValueError) as exc:
        record["issues"].append(f"invalid_feature_path: {exc}")
        return record
    try:
        expected_frames = int(row["num_frames"])
        expected_dimension = int(row["feature_dim"])
    except (TypeError, ValueError):
        record["issues"].append("manifest_frame_or_dimension_not_integer")
        return record
    record["expected_frames"] = expected_frames
    if expected_dimension != FEATURE_DIMENSION:
        record["issues"].append(
            f"manifest_feature_dim_mismatch: {expected_dimension}"
        )
    if not path.is_file():
        record["issues"].append("feature_file_missing")
        return record
    try:
        features = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        record["issues"].append(f"feature_load_error: {exc}")
        return record

    record.update(
        {
            "shape": list(features.shape),
            "dtype": str(features.dtype),
        }
    )
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        record["issues"].append(
            f"feature_shape_invalid: expected (T, {FEATURE_DIMENSION}), found {features.shape}"
        )
    elif features.shape[0] != expected_frames:
        record["issues"].append(
            f"frame_count_mismatch: manifest={expected_frames}, file={features.shape[0]}"
        )
    if features.dtype != np.float32:
        record["issues"].append(f"feature_dtype_invalid: {features.dtype}")
    if not np.issubdtype(features.dtype, np.number):
        record["issues"].append("feature_dtype_not_numeric; finite check unavailable")
    elif not np.isfinite(features).all():
        record["issues"].append("feature_contains_nan_or_inf")

    presence_stats: dict[str, float] = {}
    if features.ndim == 2 and features.shape[1] == FEATURE_DIMENSION:
        presence_values = features[:, 225:228]
        if not np.isin(presence_values, (0.0, 1.0)).all():
            record["issues"].append("presence_features_contain_values_other_than_0_or_1")
        for name, column in PRESENCE_COLUMNS:
            presence_stats[name] = (
                float(np.mean(features[:, column]) * 100.0)
                if features.shape[0] > 0
                and np.isfinite(features[:, column]).all()
                else 0.0
            )
        presence_stats["both_hands_presence"] = (
            float(np.mean(features[:, 225] * features[:, 226]) * 100.0)
            if features.shape[0] > 0
            and np.isfinite(features[:, 225:227]).all()
            else 0.0
        )
    record["presence_percentages"] = presence_stats
    record["valid"] = not record["issues"]
    return record


def main() -> int:
    expected_counts = load_split_counts()
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Feature manifest not found: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"{MANIFEST_PATH} is missing columns: {sorted(missing_columns)}"
        )

    records = [
        inspect_row(index + 2, row)
        for index, (_, row) in enumerate(manifest.iterrows())
    ]
    for index, row in manifest.iterrows():
        if str(row["split"]) not in SPLITS:
            records[index]["issues"].append(
                f"unknown_split: {row['split']!r}"
            )
            records[index]["valid"] = False

    duplicate_paths = manifest["feature_path"].duplicated(keep=False)
    for index in manifest.index[duplicate_paths]:
        records[int(index)]["issues"].append("duplicate_manifest_feature_path")
        records[int(index)]["valid"] = False

    actual_counts = {
        split: int((manifest["split"] == split).sum())
        for split in SPLITS
    }
    count_mismatches = {
        split: {
            "expected": expected_counts[split],
            "manifest": actual_counts[split],
        }
        for split in SPLITS
        if expected_counts[split] != actual_counts[split]
    }
    membership_mismatches: dict[str, dict[str, list[str]]] = {}
    for split in SPLITS:
        split_frame = pd.read_csv(SPLIT_DIR / f"{split}.csv", keep_default_na=False)
        expected_videos = {
            str(value).strip().replace("\\", "/").casefold()
            for value in split_frame["video_path"]
        }
        actual_videos = {
            str(value).strip().replace("\\", "/").casefold()
            for value in manifest.loc[manifest["split"] == split, "video_path"]
        }
        missing_videos = sorted(expected_videos - actual_videos)
        unexpected_videos = sorted(actual_videos - expected_videos)
        if missing_videos or unexpected_videos:
            membership_mismatches[split] = {
                "missing_videos": missing_videos,
                "unexpected_videos": unexpected_videos,
            }
    invalid_records = [record for record in records if not record["valid"]]
    extra_npy_files = sorted(
        path.name for path in FEATURE_DIR.glob("*.npy")
        if path.stem not in {
            Path(str(value).replace("\\", "/")).stem
            for value in manifest["feature_path"]
        }
    )
    invalid_count = (
        len(invalid_records)
        + len(extra_npy_files)
        + len(count_mismatches)
        + len(membership_mismatches)
    )

    frame_counts: dict[str, int] = {}
    presence_totals = {
        name: 0.0
        for name in (
            "left_hand_presence",
            "right_hand_presence",
            "both_hands_presence",
            "pose_presence",
        )
    }
    valid_records = [
        record for record in records
        if record["valid"] and record.get("presence_percentages")
    ]
    for record in valid_records:
        frame_count = int(record["shape"][0])
        frame_counts[record["split"]] = (
            frame_counts.get(record["split"], 0) + frame_count
        )
        for name in presence_totals:
            presence_totals[name] += (
                record["presence_percentages"][name] * frame_count / 100.0
            )
    total_valid_frames = sum(frame_counts.values())
    dataset_statistics = {
        name: (
            presence_totals[name] / total_valid_frames * 100.0
            if total_valid_frames
            else None
        )
        for name in presence_totals
    }

    report = {
        "expected_split_counts": expected_counts,
        "manifest_split_counts": actual_counts,
        "total_manifest_features": int(len(manifest)),
        "total_valid_features": len(valid_records),
        "invalid_files": invalid_count,
        "count_mismatches": count_mismatches,
        "split_membership_mismatches": membership_mismatches,
        "extra_feature_files": extra_npy_files,
        "feature_dimension": FEATURE_DIMENSION,
        "dtype_required": "float32",
        "total_valid_frames": total_valid_frames,
        "frame_weighted_presence_percentages": dataset_statistics,
        "files": records,
    }

    print("PRESENCE-AWARE FEATURE CHECK")
    print(f"Train count: {actual_counts['train']} / expected {expected_counts['train']}")
    print(f"Validation count: {actual_counts['val']} / expected {expected_counts['val']}")
    print(f"Test count: {actual_counts['test']} / expected {expected_counts['test']}")
    print(f"Total count: {len(manifest)}")
    print(f"Invalid files: {invalid_count}")
    print(f"Feature dimension: {FEATURE_DIMENSION}")
    print("Required dtype: float32")
    print(f"Total valid frames: {total_valid_frames}")
    for name in (
        "left_hand_presence",
        "right_hand_presence",
        "both_hands_presence",
        "pose_presence",
    ):
        value = dataset_statistics[name]
        print(f"{name.replace('_', ' ').title()}: {value:.2f}%" if value is not None else f"{name}: unavailable")
    if extra_npy_files:
        print(f"Unmanifested feature files: {len(extra_npy_files)}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 1 if invalid_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
