"""Analyze per-class dataset counts and split coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"
CSV_PATH = REPORT_DIR / "class_support_analysis.csv"
JSON_PATH = REPORT_DIR / "class_support_analysis.json"
SPLITS = ("train", "val", "test")
CLASS_COLUMNS = {"label_id", "class_name", "sentence", "video_path"}


def normalize_video_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    missing_columns = CLASS_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"{MANIFEST_PATH} is missing columns: {sorted(missing_columns)}"
        )
    if manifest["video_path"].map(normalize_video_path).duplicated().any():
        raise ValueError(f"{MANIFEST_PATH} contains duplicate video_path values.")

    split_frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = SPLIT_DIR / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Split file not found: {path}")
        frame = pd.read_csv(path, keep_default_na=False)
        missing = CLASS_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["video_path"].map(normalize_video_path).duplicated().any():
            raise ValueError(f"{path} contains duplicate video_path values.")
        split_frames[split] = frame
    return manifest, split_frames


def build_class_records(
    manifest: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    class_info: dict[str, dict[str, str]] = {}
    for _, row in manifest.iterrows():
        label = str(row["label_id"]).strip()
        info = {
            "class_name": str(row["class_name"]).strip(),
            "sentence": str(row["sentence"]).strip(),
        }
        existing = class_info.get(label)
        if existing is not None and existing != info:
            raise ValueError(
                f"Inconsistent class metadata for {label}: {existing} vs {info}"
            )
        class_info[label] = info

    manifest_by_video = {
        normalize_video_path(row["video_path"]): str(row["label_id"]).strip()
        for _, row in manifest.iterrows()
    }
    split_labels: dict[str, set[str]] = {}
    split_video_keys: dict[str, set[str]] = {}
    inconsistent_labels: list[dict[str, str]] = []
    unmatched_videos: dict[str, list[str]] = {}
    for split, frame in split_frames.items():
        labels: set[str] = set()
        video_keys: set[str] = set()
        unmatched: list[str] = []
        for _, row in frame.iterrows():
            video_key = normalize_video_path(row["video_path"])
            label = str(row["label_id"]).strip()
            video_keys.add(video_key)
            labels.add(label)
            manifest_label = manifest_by_video.get(video_key)
            if manifest_label is None:
                unmatched.append(str(row["video_path"]))
            elif manifest_label != label:
                inconsistent_labels.append(
                    {
                        "split": split,
                        "video_path": str(row["video_path"]),
                        "manifest_label": manifest_label,
                        "split_label": label,
                    }
                )
        split_labels[split] = labels
        split_video_keys[split] = video_keys
        unmatched_videos[split] = unmatched

    all_split_video_keys = set().union(*split_video_keys.values())
    manifest_video_keys = set(manifest_by_video)
    missing_from_splits = sorted(manifest_video_keys - all_split_video_keys)
    duplicated_across_splits = sorted(
        key
        for key in manifest_video_keys
        if sum(key in keys for keys in split_video_keys.values()) > 1
    )
    counts = {
        label: {
            "total_count": 0,
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
        }
        for label in class_info
    }
    for _, row in manifest.iterrows():
        label = str(row["label_id"]).strip()
        counts[label]["total_count"] += 1
    for split, frame in split_frames.items():
        count_key = {"train": "train_count", "val": "val_count", "test": "test_count"}[
            split
        ]
        for label, count in frame["label_id"].astype(str).value_counts().items():
            label = str(label).strip()
            if label not in counts:
                raise ValueError(
                    f"{split} includes class {label!r} absent from the manifest."
                )
            counts[label][count_key] = int(count)

    records = [
        {
            "class_id": label,
            "class_name": class_info[label]["class_name"],
            "total_count": counts[label]["total_count"],
            "train_count": counts[label]["train_count"],
            "val_count": counts[label]["val_count"],
            "test_count": counts[label]["test_count"],
        }
        for label in sorted(class_info)
    ]
    total_counts = [record["total_count"] for record in records]
    buckets = {
        str(count): sum(total == count for total in total_counts)
        for count in range(1, 9)
    }
    buckets["9+"] = sum(total >= 9 for total in total_counts)
    represented = {
        split: len(labels) for split, labels in split_labels.items()
    }
    missing_validation = [
        record for record in records if record["val_count"] == 0
    ]
    missing_test = [
        record for record in records if record["test_count"] == 0
    ]
    fewer_total = [record for record in records if record["total_count"] < 5]
    fewer_train = [record for record in records if record["train_count"] < 5]
    fewer_val_test = [
        record
        for record in records
        if record["val_count"] < 2 or record["test_count"] < 2
    ]
    summary = {
        "total_videos_in_manifest": int(len(manifest)),
        "total_classes": len(records),
        "total_videos_by_split": {
            split: int(len(frame)) for split, frame in split_frames.items()
        },
        "classes_by_total_video_count": buckets,
        "classes_represented_by_split": represented,
        "classes_missing_from_validation": missing_validation,
        "classes_missing_from_test": missing_test,
        "classes_with_fewer_than_5_total_videos": fewer_total,
        "classes_with_fewer_than_5_training_videos": fewer_train,
        "classes_with_fewer_than_2_validation_or_test_samples": fewer_val_test,
        "classes_with_fewer_than_2_validation_samples": [
            record for record in records if record["val_count"] < 2
        ],
        "classes_with_fewer_than_2_test_samples": [
            record for record in records if record["test_count"] < 2
        ],
        "total_video_count_statistics": {
            "mean": float(np.mean(total_counts)) if total_counts else None,
            "median": float(np.median(total_counts)) if total_counts else None,
            "minimum": int(min(total_counts)) if total_counts else None,
            "maximum": int(max(total_counts)) if total_counts else None,
        },
        "split_consistency": {
            "unmatched_split_videos": unmatched_videos,
            "videos_in_manifest_missing_from_all_splits": missing_from_splits,
            "videos_duplicated_across_splits": duplicated_across_splits,
            "label_mismatches": inconsistent_labels,
        },
    }
    return records, summary


def print_class_list(title: str, records: list[dict[str, Any]]) -> None:
    print(f"\n{title}: {len(records)}")
    if not records:
        print("  None")
        return
    for record in records:
        print(
            f"  {record['class_id']} | {record['class_name']} | "
            f"total={record['total_count']} train={record['train_count']} "
            f"val={record['val_count']} test={record['test_count']}"
        )


def main() -> int:
    manifest, split_frames = load_inputs()
    records, summary = build_class_records(manifest, split_frames)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    JSON_PATH.write_text(
        json.dumps(
            {"summary": summary, "classes": records},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("ISL BRIDGE CLASS SUPPORT ANALYSIS")
    print(f"Manifest videos: {summary['total_videos_in_manifest']}")
    print(f"Classes: {summary['total_classes']}")
    print(f"Videos by split: {summary['total_videos_by_split']}")
    print(f"Class count buckets (total videos/class): {summary['classes_by_total_video_count']}")
    print(f"Classes represented by split: {summary['classes_represented_by_split']}")
    stats = summary["total_video_count_statistics"]
    print(
        "Videos/class statistics: "
        f"mean={stats['mean']:.3f} median={stats['median']:.3f} "
        f"minimum={stats['minimum']} maximum={stats['maximum']}"
    )
    print_class_list(
        "CLASSES MISSING FROM VALIDATION",
        summary["classes_missing_from_validation"],
    )
    print_class_list("CLASSES MISSING FROM TEST", summary["classes_missing_from_test"])
    print_class_list(
        "CLASSES WITH FEWER THAN 5 TOTAL VIDEOS",
        summary["classes_with_fewer_than_5_total_videos"],
    )
    print_class_list(
        "CLASSES WITH FEWER THAN 5 TRAINING VIDEOS",
        summary["classes_with_fewer_than_5_training_videos"],
    )
    print_class_list(
        "CLASSES WITH FEWER THAN 2 VALIDATION OR TEST SAMPLES",
        summary["classes_with_fewer_than_2_validation_or_test_samples"],
    )
    print("\nPER-CLASS SUPPORT")
    for record in records:
        print(
            f"{record['class_id']} | {record['class_name']} | "
            f"total={record['total_count']} train={record['train_count']} "
            f"val={record['val_count']} test={record['test_count']}"
        )
    print("\nSPLIT CONSISTENCY")
    consistency = summary["split_consistency"]
    print(
        "Unmatched videos by split: "
        + str({split: len(paths) for split, paths in consistency["unmatched_split_videos"].items()})
    )
    print(
        f"Manifest videos missing from splits: "
        f"{len(consistency['videos_in_manifest_missing_from_all_splits'])}"
    )
    print(f"Videos duplicated across splits: {len(consistency['videos_duplicated_across_splits'])}")
    print(f"Label mismatches: {len(consistency['label_mismatches'])}")
    print(f"\nCSV report: {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON report: {JSON_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
