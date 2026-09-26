"""Summarize V2 per-frame detector presence for training videos and classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "landmark_metadata_v2"
REPORT_PATH = PROJECT_ROOT / "reports" / "v2_presence_analysis.json"
SEED = 42
HAND_IMBALANCE_THRESHOLD_PERCENT = 10.0
PRESENCE_KEYS = {
    "left_hand_presence_percent": "left_hand_present",
    "right_hand_presence_percent": "right_hand_present",
    "both_hands_presence_percent": None,
    "pose_presence_percent": "pose_present",
}


def metadata_path_for(row: pd.Series) -> Path:
    landmark_path = Path(str(row["landmark_path"]).replace("\\", "/"))
    return METADATA_DIR / f"{landmark_path.stem}.npz"


def inspect_video(row_index: int, row: pd.Series) -> dict[str, Any]:
    path = metadata_path_for(row)
    record: dict[str, Any] = {
        "training_row": row_index + 2,
        "landmark_filename": Path(str(row["landmark_path"])).name,
        "metadata_filename": path.name,
        "class_name": str(row["class_name"]).strip(),
        "label_id": str(row.get("label_id", "")).strip(),
        "sentence": str(row.get("sentence", "")).strip(),
        "video_path": str(row.get("video_path", "")).strip(),
        "valid": False,
        "frame_count": 0,
        "error": None,
    }
    if not path.is_file():
        record["error"] = "metadata_file_missing"
        return record

    try:
        with np.load(path, allow_pickle=False) as metadata:
            missing_keys = [
                key for key in ("left_hand_present", "right_hand_present", "pose_present")
                if key not in metadata.files
            ]
            if missing_keys:
                raise ValueError(f"missing_presence_arrays: {missing_keys}")

            arrays = {
                key: metadata[key]
                for key in ("left_hand_present", "right_hand_present", "pose_present")
            }
    except (OSError, ValueError) as exc:
        record["error"] = f"metadata_read_error: {exc}"
        return record

    lengths = {key: len(values) for key, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        record["error"] = f"presence_array_length_mismatch: {lengths}"
        return record
    invalid_dtypes = {
        key: str(values.dtype)
        for key, values in arrays.items()
        if values.dtype != np.bool_
    }
    if invalid_dtypes:
        record["error"] = f"presence_arrays_not_boolean: {invalid_dtypes}"
        return record
    frame_count = next(iter(lengths.values()), 0)
    if frame_count == 0:
        record["error"] = "empty_presence_arrays"
        return record

    left = arrays["left_hand_present"]
    right = arrays["right_hand_present"]
    pose = arrays["pose_present"]
    record.update(
        {
            "valid": True,
            "frame_count": frame_count,
            "left_hand_presence_percent": float(np.mean(left) * 100.0),
            "right_hand_presence_percent": float(np.mean(right) * 100.0),
            "both_hands_presence_percent": float(np.mean(left & right) * 100.0),
            "pose_presence_percent": float(np.mean(pose) * 100.0),
        }
    )
    return record


def aggregate_classes(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(videos)
    metrics = (
        "left_hand_presence_percent",
        "right_hand_presence_percent",
        "both_hands_presence_percent",
        "pose_presence_percent",
    )
    classes: list[dict[str, Any]] = []
    for class_name, group in frame.groupby("class_name", sort=True):
        result: dict[str, Any] = {
            "class_name": str(class_name),
            "number_of_videos": int(len(group)),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            result[f"mean_{metric}"] = float(np.mean(values))
            result[f"std_{metric}"] = float(np.std(values, ddof=0))
        result["minimum_both_hand_presence_percent"] = float(
            group["both_hands_presence_percent"].min()
        )
        result["maximum_both_hand_presence_percent"] = float(
            group["both_hands_presence_percent"].max()
        )
        result["left_minus_right_presence_percentage_points"] = (
            result["mean_left_hand_presence_percent"]
            - result["mean_right_hand_presence_percent"]
        )
        classes.append(result)
    return classes


def class_order(classes: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return sorted(
        classes,
        key=lambda record: (record[metric], record["class_name"]),
    )


def video_order(videos: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return sorted(
        videos,
        key=lambda record: (record[metric], record["class_name"], record["metadata_filename"]),
    )


def rounded_records(
    records: list[dict[str, Any]],
    numeric_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        copy = dict(record)
        for key in numeric_keys:
            if key in copy:
                copy[key] = round(float(copy[key]), 4)
        output.append(copy)
    return output


def print_class_table(title: str, records: list[dict[str, Any]], metric: str) -> None:
    print(f"\n{title}")
    if not records:
        print("  No valid class records.")
        return
    for rank, record in enumerate(records[:20], start=1):
        print(
            f"{rank:2}. {record['class_name']} | "
            f"videos={record['number_of_videos']} | "
            f"mean both hands={record[metric]:.2f}% | "
            f"std={record['std_both_hands_presence_percent']:.2f}% | "
            f"range={record['minimum_both_hand_presence_percent']:.2f}-"
            f"{record['maximum_both_hand_presence_percent']:.2f}%"
        )


def print_video_table(title: str, records: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    if not records:
        print("  No valid video records.")
        return
    for rank, record in enumerate(records[:20], start=1):
        print(
            f"{rank:2}. {record['metadata_filename']} | {record['class_name']} | "
            f"both hands={record['both_hands_presence_percent']:.2f}% | "
            f"left={record['left_hand_presence_percent']:.2f}% | "
            f"right={record['right_hand_presence_percent']:.2f}%"
        )


def main() -> int:
    np.random.seed(SEED)
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Training split not found: {TRAIN_PATH}")
    if not METADATA_DIR.is_dir():
        raise FileNotFoundError(f"V2 metadata directory not found: {METADATA_DIR}")

    training = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    required_columns = {"landmark_path", "class_name"}
    missing_columns = required_columns - set(training.columns)
    if missing_columns:
        raise ValueError(
            f"{TRAIN_PATH} is missing required columns: {sorted(missing_columns)}"
        )
    records = [
        inspect_video(index, row)
        for index, (_, row) in enumerate(training.iterrows())
    ]
    valid_videos = [record for record in records if record["valid"]]
    invalid_videos = [record for record in records if not record["valid"]]
    if not valid_videos:
        raise ValueError(
            "No valid V2 presence metadata matched the training split. "
            "Run scripts/extract_landmarks_v2.py for a selected train subset first."
        )

    classes = aggregate_classes(valid_videos)
    lowest_classes = class_order(
        classes, "mean_both_hands_presence_percent"
    )
    highest_classes = list(reversed(class_order(
        classes, "mean_both_hands_presence_percent"
    )))
    highest_variability = sorted(
        classes,
        key=lambda item: (
            -item["std_both_hands_presence_percent"],
            item["class_name"],
        ),
    )
    lowest_videos = video_order(valid_videos, "both_hands_presence_percent")
    highest_videos = list(reversed(lowest_videos))

    left_much_lower = sorted(
        [
            item
            for item in classes
            if item["left_minus_right_presence_percentage_points"]
            <= -HAND_IMBALANCE_THRESHOLD_PERCENT
        ],
        key=lambda item: (
            item["left_minus_right_presence_percentage_points"],
            item["class_name"],
        ),
    )
    right_much_lower = sorted(
        [
            item
            for item in classes
            if item["left_minus_right_presence_percentage_points"]
            >= HAND_IMBALANCE_THRESHOLD_PERCENT
        ],
        key=lambda item: (
            -item["left_minus_right_presence_percentage_points"],
            item["class_name"],
        ),
    )

    metrics = tuple(PRESENCE_KEYS)
    dataset_wide = {
        f"mean_{metric}": float(np.mean([record[metric] for record in valid_videos]))
        for metric in metrics
    }
    total_frames = sum(record["frame_count"] for record in valid_videos)
    frame_weighted = {
        f"frame_weighted_{metric}": (
            sum(
                record[metric] * record["frame_count"]
                for record in valid_videos
            )
            / total_frames
        )
        for metric in metrics
    }
    class_both_means = [
        item["mean_both_hands_presence_percent"] for item in classes
    ]
    class_summary = {
        "mean_class_mean_both_hand_presence_percent": float(
            np.mean(class_both_means)
        ),
        "std_class_mean_both_hand_presence_percent": float(
            np.std(class_both_means, ddof=0)
        ),
    }
    lowest_class = lowest_classes[0]
    highest_class = highest_classes[0]

    report = {
        "seed": SEED,
        "inputs": {
            "training_split": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
            "metadata_directory": str(METADATA_DIR.relative_to(PROJECT_ROOT)),
        },
        "selection_notes": {
            "presence_percentages_are_computed_per_video_from_frame_boolean_arrays": True,
            "dataset_wide_means_are_unweighted_means_across_valid_videos": True,
            "frame_weighted_presence_is_also_reported": True,
            "class_statistics_use_population_standard_deviation_ddof_0": True,
            "hand_imbalance_threshold_percentage_points": HAND_IMBALANCE_THRESHOLD_PERCENT,
            "hand_imbalance_rule": (
                "A hand is described as much lower when its class mean presence "
                "is at least 10 percentage points below the other hand."
            ),
        },
        "dataset_summary": {
            "total_training_videos": int(len(training)),
            "total_valid_videos": int(len(valid_videos)),
            "videos_missing_or_with_invalid_metadata": int(len(invalid_videos)),
            "total_classes_in_training_split": int(training["class_name"].nunique()),
            "classes_with_valid_metadata": int(len(classes)),
            "total_frames_in_valid_metadata": int(total_frames),
            **dataset_wide,
            **frame_weighted,
            **class_summary,
            "lowest_class": lowest_class,
            "highest_class": highest_class,
        },
        "individual_video_statistics": records,
        "class_statistics": classes,
        "lowest_mean_both_hand_presence_classes": lowest_classes[:20],
        "highest_mean_both_hand_presence_classes": highest_classes[:20],
        "highest_within_class_both_hand_variation": highest_variability[:20],
        "lowest_both_hand_presence_videos": lowest_videos[:20],
        "highest_both_hand_presence_videos": highest_videos[:20],
        "left_hand_much_lower_classes": left_much_lower,
        "right_hand_much_lower_classes": right_much_lower,
        "invalid_or_missing_videos": invalid_videos,
    }

    print("V2 PRESENCE ANALYSIS")
    print(f"Training split videos: {len(training)}")
    print(f"Valid metadata videos: {len(valid_videos)}")
    print(f"Missing/invalid metadata videos: {len(invalid_videos)}")
    print(f"Classes represented by valid metadata: {len(classes)}")
    print("Dataset-wide (unweighted mean across valid videos):")
    for metric in metrics:
        print(f"  {metric.replace('_percent', '').replace('_', ' ')}: {dataset_wide[f'mean_{metric}']:.2f}%")
    print("Frame-weighted:")
    for metric in metrics:
        print(f"  {metric.replace('_percent', '').replace('_', ' ')}: {frame_weighted[f'frame_weighted_{metric}']:.2f}%")

    print_class_table(
        "20 CLASSES WITH LOWEST MEAN BOTH-HAND PRESENCE",
        lowest_classes,
        "mean_both_hands_presence_percent",
    )
    print_class_table(
        "20 CLASSES WITH HIGHEST MEAN BOTH-HAND PRESENCE",
        highest_classes,
        "mean_both_hands_presence_percent",
    )
    print_class_table(
        "20 CLASSES WITH HIGHEST WITHIN-CLASS VARIATION",
        highest_variability,
        "mean_both_hands_presence_percent",
    )
    print_video_table("20 VIDEOS WITH LOWEST BOTH-HAND PRESENCE", lowest_videos)
    print_video_table("20 VIDEOS WITH HIGHEST BOTH-HAND PRESENCE", highest_videos)

    print(
        "\nHAND-PRESENCE DIFFERENCES "
        f"(threshold: {HAND_IMBALANCE_THRESHOLD_PERCENT:.1f} percentage points)"
    )
    for title, group in (
        ("Left-hand presence much lower than right-hand presence", left_much_lower),
        ("Right-hand presence much lower than left-hand presence", right_much_lower),
    ):
        print(f"{title}:")
        if not group:
            print("  None in the valid metadata subset.")
        for item in group:
            difference = item["left_minus_right_presence_percentage_points"]
            print(
                f"  {item['class_name']}: left-right={difference:+.2f} pp "
                f"(left={item['mean_left_hand_presence_percent']:.2f}%, "
                f"right={item['mean_right_hand_presence_percent']:.2f}%)"
            )

    print("\nV2 PRESENCE ANALYSIS SUMMARY")
    print(f"Total videos: {len(valid_videos)} valid / {len(training)} in train split")
    print(f"Total classes: {len(classes)} with valid metadata")
    print(f"Mean left-hand presence: {dataset_wide['mean_left_hand_presence_percent']:.2f}%")
    print(f"Mean right-hand presence: {dataset_wide['mean_right_hand_presence_percent']:.2f}%")
    print(f"Mean both-hand presence: {dataset_wide['mean_both_hands_presence_percent']:.2f}%")
    print(
        "Mean/std class-level both-hand presence: "
        f"{class_summary['mean_class_mean_both_hand_presence_percent']:.2f}% / "
        f"{class_summary['std_class_mean_both_hand_presence_percent']:.2f}%"
    )
    print(
        f"Lowest class: {lowest_class['class_name']} "
        f"({lowest_class['mean_both_hands_presence_percent']:.2f}%)"
    )
    print(
        f"Highest class: {highest_class['class_name']} "
        f"({highest_class['mean_both_hands_presence_percent']:.2f}%)"
    )
    print(
        "Neutral observation: these statistics describe the available V2 train "
        "metadata only; missing or invalid metadata is listed in the JSON report."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nReport: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
