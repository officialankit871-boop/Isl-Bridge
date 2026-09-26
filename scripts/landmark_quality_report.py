"""Generate a training-split quality report for extracted ISL landmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "landmark_quality_report.json"
FEATURE_DIMENSION = 225
LANDMARK_COUNT = 75
SEED = 42

COMPONENTS: dict[str, tuple[int, int]] = {
    "POSE": (0, 99),
    "LEFT_HAND": (99, 162),
    "RIGHT_HAND": (162, 225),
    "BOTH_HANDS": (99, 225),
}
PRESENCE_COMPONENTS = ("POSE", "LEFT_HAND", "RIGHT_HAND", "BOTH_HANDS")
PERCENTILES = (5, 25, 50, 75, 90, 95, 99)


def resolve_path(value: str) -> Path:
    """Resolve split paths against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def finite_frame_mask(values: np.ndarray) -> np.ndarray:
    """Return true for frames whose 225 values are all finite."""
    return np.isfinite(values).all(axis=1)


def component_presence(values: np.ndarray, start: int, end: int) -> np.ndarray:
    """Determine component presence using the all-zero landmark contract."""
    component = values[:, start:end]
    valid = np.isfinite(component).all(axis=(1, 2))
    return valid & np.any(component != 0.0, axis=(1, 2))


def transition_magnitudes(
    values: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Compute finite frame-to-frame Euclidean displacement magnitudes."""
    if values.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    component = values[:, start:end, :]
    valid_pairs = np.isfinite(component[:-1]).all(axis=(1, 2)) & np.isfinite(
        component[1:]
    ).all(axis=(1, 2))
    if not valid_pairs.any():
        return np.empty(0, dtype=np.float64)
    differences = component[1:][valid_pairs] - component[:-1][valid_pairs]
    return np.linalg.norm(differences.reshape(len(differences), -1), axis=1)


def distribution_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    """Summarize a finite numeric distribution, including requested percentiles."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "percentiles": {str(percentile): None for percentile in PERCENTILES},
        }
    percentile_values = np.percentile(array, PERCENTILES)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "percentiles": {
            str(percentile): float(value)
            for percentile, value in zip(PERCENTILES, percentile_values)
        },
    }


def analyze_landmark_file(
    path: Path,
    metadata: dict[str, str],
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Calculate per-video presence, zeros, motion, jitter, and validation."""
    result: dict[str, Any] = {
        **metadata,
        "landmark_path": str(path),
        "valid_file": False,
        "shape": None,
        "dtype": None,
        "shape_valid": False,
        "dtype_valid": False,
        "frames": None,
        "pose_presence_ratio": None,
        "left_hand_presence_ratio": None,
        "right_hand_presence_ratio": None,
        "both_hands_present_ratio": None,
        "zero_coordinate_percentage": None,
        "nan_inf_frame_percentage": None,
        "nan_inf_coordinate_percentage": None,
        "nearly_static_frame_percentage": None,
        "movement": {},
        "temporal_jitter": None,
        "valid_transition_counts": {},
        "component_zero_counts": {},
        "component_coordinate_counts": {},
        "error": None,
    }
    movement_samples: dict[str, list[float]] = {
        name: [] for name in COMPONENTS
    }

    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError, EOFError) as exc:
        result["error"] = f"Could not load landmark file: {exc}"
        return result, movement_samples

    result["shape"] = list(values.shape)
    result["dtype"] = str(values.dtype)
    result["shape_valid"] = bool(
        values.ndim == 2 and values.shape[0] > 0 and values.shape[1] == FEATURE_DIMENSION
    )
    result["dtype_valid"] = bool(values.dtype == np.float32)
    if not result["shape_valid"]:
        result["error"] = (
            f"Expected non-empty shape (T, {FEATURE_DIMENSION}), got {values.shape}."
        )
        return result, movement_samples
    if not np.issubdtype(values.dtype, np.number):
        result["error"] = f"Expected numeric landmark dtype, got {values.dtype}."
        return result, movement_samples

    values = np.asarray(values, dtype=np.float64).reshape(-1, LANDMARK_COUNT, 3)
    frame_count = int(values.shape[0])
    invalid_frame_mask = ~finite_frame_mask(values.reshape(frame_count, FEATURE_DIMENSION))
    invalid_coordinate_mask = ~np.isfinite(values)
    invalid_coordinate_count = int(invalid_coordinate_mask.sum())
    total_coordinate_count = int(values.size)
    result["valid_file"] = True
    result["frames"] = frame_count
    result["nan_inf_frame_count"] = int(invalid_frame_mask.sum())
    result["nan_inf_frame_percentage"] = float(
        100.0 * invalid_frame_mask.mean()
    )
    result["nan_inf_coordinate_count"] = invalid_coordinate_count
    result["nan_inf_coordinate_percentage"] = (
        100.0 * invalid_coordinate_count / total_coordinate_count
        if total_coordinate_count
        else None
    )
    zero_count = int(np.sum(values == 0.0))
    result["zero_coordinate_count"] = zero_count
    result["coordinate_count"] = total_coordinate_count
    result["zero_coordinate_percentage"] = (
        100.0 * zero_count / total_coordinate_count
        if total_coordinate_count
        else None
    )
    for component_name, (start, end) in COMPONENTS.items():
        component_values = values[:, start // 3 : end // 3, :]
        result["component_zero_counts"][component_name] = int(
            np.sum(component_values == 0.0)
        )
        result["component_coordinate_counts"][component_name] = int(
            component_values.size
        )

    presence: dict[str, np.ndarray] = {}
    for component_name, (start, end) in COMPONENTS.items():
        presence[component_name] = component_presence(values, start // 3, end // 3)
    result["pose_presence_ratio"] = float(presence["POSE"].mean())
    result["left_hand_presence_ratio"] = float(presence["LEFT_HAND"].mean())
    result["right_hand_presence_ratio"] = float(presence["RIGHT_HAND"].mean())
    result["both_hands_present_ratio"] = float(
        np.logical_and(presence["LEFT_HAND"], presence["RIGHT_HAND"]).mean()
    )

    for component_name, (start, end) in COMPONENTS.items():
        movement = transition_magnitudes(values, start // 3, end // 3)
        movement_samples[component_name] = movement.tolist()
        result["movement"][component_name] = distribution_summary(movement)
        result["valid_transition_counts"][component_name] = int(movement.size)

    hand_motion = movement_samples["BOTH_HANDS"]
    jitter = distribution_summary(hand_motion)
    result["temporal_jitter"] = {
        "definition": (
            "Distribution of consecutive-frame BOTH_HANDS Euclidean displacement "
            "magnitudes; invalid frame pairs are excluded."
        ),
        "mean": jitter["mean"],
        "std": jitter["std"],
        "median": jitter["median"],
        "p95": jitter["percentiles"]["95"],
        "valid_transition_count": jitter["count"],
    }
    result["nearly_static_transition_count"] = 0
    return result, movement_samples


def aggregate_video_metric(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Summarize a per-video scalar across valid analyzed files."""
    return distribution_summary(
        [
            float(record[key])
            for record in records
            if record.get("valid_file") and record.get(key) is not None
        ]
    )


def derive_static_threshold(all_hand_movement: list[float]) -> float | None:
    """Use the empirical 5th percentile of hand transitions as a data-derived cutoff."""
    values = np.asarray(all_hand_movement, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return float(np.percentile(values, 5))


def apply_static_frame_metrics(
    records: list[dict[str, Any]],
    threshold: float | None,
) -> None:
    """Compute per-video nearly-static transition percentages at one global cutoff."""
    for record in records:
        if not record.get("valid_file"):
            continue
        values = np.asarray(record["movement_samples"]["BOTH_HANDS"], dtype=np.float64)
        if threshold is None or values.size == 0:
            record["nearly_static_transition_count"] = 0
            record["nearly_static_frame_percentage"] = None
            continue
        static_count = int(np.sum(values <= threshold))
        record["nearly_static_transition_count"] = static_count
        record["nearly_static_frame_percentage"] = float(
            100.0 * static_count / values.size
        )


def per_class_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute class-level mean and standard deviation of requested measures."""
    valid_records = [record for record in records if record.get("valid_file")]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in valid_records:
        grouped.setdefault(record["class_name"], []).append(record)

    output: list[dict[str, Any]] = []
    metric_keys = {
        "left_hand_presence_ratio": "left_hand_presence_ratio",
        "right_hand_presence_ratio": "right_hand_presence_ratio",
        "both_hand_presence_ratio": "both_hands_present_ratio",
        "hand_movement": None,
        "pose_movement": None,
        "zero_coordinate_percentage": "zero_coordinate_percentage",
    }
    for class_name in sorted(grouped):
        class_records = grouped[class_name]
        class_result: dict[str, Any] = {
            "class_name": class_name,
            "class_id": class_records[0].get("class_id"),
            "sentence": class_records[0].get("sentence"),
            "number_of_videos": len(class_records),
        }
        for output_key, record_key in metric_keys.items():
            if record_key is None:
                component_name = "BOTH_HANDS" if output_key == "hand_movement" else "POSE"
                values = [
                    record["movement"][component_name]["mean"]
                    for record in class_records
                    if record["movement"][component_name]["mean"] is not None
                ]
            else:
                values = [
                    record[record_key]
                    for record in class_records
                    if record.get(record_key) is not None
                ]
            summary = distribution_summary(values)
            class_result[output_key] = {
                "mean": summary["mean"],
                "std": summary["std"],
                "count": summary["count"],
            }
        output.append(class_result)
    return output


def top_records(
    records: list[dict[str, Any]],
    metric: str,
    count: int,
    reverse: bool,
) -> list[dict[str, Any]]:
    """Sort records by a scalar metric, excluding invalid or unavailable values."""
    candidates = [
        record
        for record in records
        if record.get("valid_file") and record.get(metric) is not None
    ]
    candidates.sort(
        key=lambda record: (
            float(record[metric]),
            record.get("class_id", ""),
            record.get("landmark_path", ""),
        ),
        reverse=reverse,
    )
    return candidates[:count]


def top_record_payload(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep useful identifying and diagnostic fields in ranked video lists."""
    return [
        {
            "rank": rank,
            "class_id": record.get("class_id"),
            "class_name": record.get("class_name"),
            "sentence": record.get("sentence"),
            "file_name": record.get("file_name"),
            "landmark_path": record["landmark_path"],
            "frames": record["frames"],
            "both_hands_present_ratio": record["both_hands_present_ratio"],
            "hand_movement_mean": record["movement"]["BOTH_HANDS"]["mean"],
            "temporal_jitter": record["temporal_jitter"],
            "zero_coordinate_percentage": record["zero_coordinate_percentage"],
            "nearly_static_frame_percentage": record.get(
                "nearly_static_frame_percentage"
            ),
        }
        for rank, record in enumerate(records, start=1)
    ]


def class_variability_report(
    class_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank classes by within-class variation across key quality metrics."""
    output: list[dict[str, Any]] = []
    for metric in (
        "left_hand_presence_ratio",
        "right_hand_presence_ratio",
        "both_hand_presence_ratio",
        "hand_movement",
        "pose_movement",
        "zero_coordinate_percentage",
    ):
        eligible = [
            record
            for record in class_records
            if record[metric]["std"] is not None
            and record[metric]["std"] > 0
            and record["number_of_videos"] > 1
        ]
        eligible.sort(
            key=lambda record: (
                -float(record[metric]["std"]),
                record["class_name"],
            )
        )
        output.extend(
            {
                "metric": metric,
                "class_name": record["class_name"],
                "class_id": record["class_id"],
                "sentence": record["sentence"],
                "number_of_videos": record["number_of_videos"],
                "mean": record[metric]["mean"],
                "std": record[metric]["std"],
            }
            for record in eligible[:10]
        )
    return output


def print_video_ranking(
    title: str,
    records: list[dict[str, Any]],
    value_key: str,
    display_name: str,
) -> None:
    """Print ranked video summaries selected for the report."""
    print(f"\n{title}")
    for rank, record in enumerate(records, start=1):
        value = record.get(value_key)
        if isinstance(value, dict):
            value = value.get("mean")
        printable = f"{value:.6f}" if value is not None else "n/a"
        print(
            f"{rank:2d}. {record.get('class_id')} | {record.get('file_name')} | "
            f"{display_name}={printable} | {record['landmark_path']}"
        )


def main() -> int:
    """Analyze all landmark rows referenced by the training split and save JSON."""
    np.random.seed(SEED)
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Training split not found: {TRAIN_PATH}")
    train_frame = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    required_columns = {"landmark_path", "class_name"}
    missing_columns = required_columns - set(train_frame.columns)
    if missing_columns:
        raise ValueError(
            f"{TRAIN_PATH} is missing required columns: {sorted(missing_columns)}"
        )

    all_records: list[dict[str, Any]] = []
    hand_movements: list[float] = []
    for row_index, row in train_frame.iterrows():
        path_value = str(row["landmark_path"]).strip()
        metadata = {
            "split_row": str(row_index + 2),
            "class_name": str(row["class_name"]).strip(),
            "class_id": str(row.get("label_id", "")).strip() or None,
            "sentence": str(row.get("sentence", "")).strip() or None,
            "file_name": str(row.get("file_name", "")).strip() or None,
        }
        path = resolve_path(path_value) if path_value else PROJECT_ROOT
        record, movement_samples = analyze_landmark_file(path, metadata)
        if record["valid_file"]:
            record["movement_samples"] = movement_samples
            hand_movements.extend(movement_samples["BOTH_HANDS"])
        else:
            record["movement_samples"] = {
                component: [] for component in COMPONENTS
            }
            record["nan_inf_frame_count"] = None
            record["nan_inf_coordinate_count"] = None
            record["zero_coordinate_count"] = None
            record["coordinate_count"] = None
            for component in COMPONENTS:
                record["component_zero_counts"][component] = 0
                record["component_coordinate_counts"][component] = 0
        all_records.append(record)
        if (row_index + 1) % 50 == 0 or row_index + 1 == len(train_frame):
            print(f"Analyzed {row_index + 1}/{len(train_frame)} training rows.")

    static_threshold = derive_static_threshold(hand_movements)
    apply_static_frame_metrics(all_records, static_threshold)
    valid_records = [record for record in all_records if record["valid_file"]]
    invalid_records = [
        record
        for record in all_records
        if not record["valid_file"] or not record["dtype_valid"]
    ]

    aggregate = {
        "frames_per_video": aggregate_video_metric(all_records, "frames"),
        "pose_presence_ratio": aggregate_video_metric(
            all_records, "pose_presence_ratio"
        ),
        "left_hand_presence_ratio": aggregate_video_metric(
            all_records, "left_hand_presence_ratio"
        ),
        "right_hand_presence_ratio": aggregate_video_metric(
            all_records, "right_hand_presence_ratio"
        ),
        "both_hands_present_ratio": aggregate_video_metric(
            all_records, "both_hands_present_ratio"
        ),
        "zero_coordinate_percentage": aggregate_video_metric(
            all_records, "zero_coordinate_percentage"
        ),
        "nan_inf_frame_percentage": aggregate_video_metric(
            all_records, "nan_inf_frame_percentage"
        ),
        "nan_inf_coordinate_percentage": aggregate_video_metric(
            all_records, "nan_inf_coordinate_percentage"
        ),
        "nearly_static_frame_percentage": aggregate_video_metric(
            all_records, "nearly_static_frame_percentage"
        ),
        "hand_movement_mean_per_video": distribution_summary(
            [
                record["movement"]["BOTH_HANDS"]["mean"]
                for record in valid_records
                if record["movement"]["BOTH_HANDS"]["mean"] is not None
            ]
        ),
        "pose_movement_mean_per_video": distribution_summary(
            [
                record["movement"]["POSE"]["mean"]
                for record in valid_records
                if record["movement"]["POSE"]["mean"] is not None
            ]
        ),
        "temporal_jitter_both_hands_all_transitions": distribution_summary(
            [
                movement
                for record in all_records
                for movement in record.get("movement_samples", {}).get(
                    "BOTH_HANDS", []
                )
            ]
        ),
        "temporal_jitter_both_hands_per_video_mean": distribution_summary(
            [
                record["temporal_jitter"]["mean"]
                for record in valid_records
                if record["temporal_jitter"]["mean"] is not None
            ]
        ),
    }

    component_aggregate: dict[str, Any] = {}
    for component_name, (start, end) in COMPONENTS.items():
        presence_key = {
            "POSE": "pose_presence_ratio",
            "LEFT_HAND": "left_hand_presence_ratio",
            "RIGHT_HAND": "right_hand_presence_ratio",
            "BOTH_HANDS": "both_hands_present_ratio",
        }[component_name]
        zero_values: list[float] = []
        movement_means: list[float] = []
        movement_all: list[float] = []
        for record in all_records:
            if not record["valid_file"]:
                continue
            coordinate_count = record["component_coordinate_counts"][component_name]
            if coordinate_count:
                zero_values.append(
                    100.0
                    * record["component_zero_counts"][component_name]
                    / coordinate_count
                )
            movement_summary = record["movement"][component_name]
            if movement_summary["mean"] is not None:
                movement_means.append(float(movement_summary["mean"]))
            movement_all.extend(record["movement_samples"][component_name])
        component_aggregate[component_name] = {
            "presence_ratio": aggregate_video_metric(all_records, presence_key),
            "zero_coordinate_percentage": distribution_summary(zero_values),
            "frame_to_frame_movement_per_video_mean": distribution_summary(
                movement_means
            ),
            "frame_to_frame_movement_all_transitions": distribution_summary(
                movement_all
            ),
        }

    for record in all_records:
        record.pop("movement_samples", None)
        record.pop("component_zero_counts", None)
        record.pop("component_coordinate_counts", None)

    class_records = per_class_summary(all_records)
    class_variability = class_variability_report(class_records)
    print("\nCLASSES WITH HIGHEST INTERNAL VARIATION (top 20 metric/class records)")
    for rank, entry in enumerate(class_variability[:20], start=1):
        print(
            f"{rank:2d}. {entry['metric']} | {entry['class_id']} "
            f"({entry['class_name']}) | mean={entry['mean']:.6f} | "
            f"std={entry['std']:.6f} | videos={entry['number_of_videos']}"
        )
    lowest_both_presence = top_records(
        all_records, "both_hands_present_ratio", 20, reverse=False
    )
    highest_both_presence = top_records(
        all_records, "both_hands_present_ratio", 20, reverse=True
    )
    lowest_hand_movement = sorted(
        (
            record
            for record in valid_records
            if record["movement"]["BOTH_HANDS"]["mean"] is not None
        ),
        key=lambda record: (
            record["movement"]["BOTH_HANDS"]["mean"],
            record.get("class_id") or "",
            record["landmark_path"],
        ),
    )[:20]
    highest_hand_movement = sorted(
        (
            record
            for record in valid_records
            if record["movement"]["BOTH_HANDS"]["mean"] is not None
        ),
        key=lambda record: (
            -record["movement"]["BOTH_HANDS"]["mean"],
            record.get("class_id") or "",
            record["landmark_path"],
        ),
    )[:20]
    highest_zero_percent = top_records(
        all_records, "zero_coordinate_percentage", 20, reverse=True
    )
    highest_jitter = sorted(
        (
            record
            for record in valid_records
            if record["temporal_jitter"]["mean"] is not None
        ),
        key=lambda record: (
            -record["temporal_jitter"]["mean"],
            record.get("class_id") or "",
            record["landmark_path"],
        ),
    )[:20]

    print_video_ranking(
        "20 VIDEOS WITH LOWEST BOTH-HAND PRESENCE",
        top_record_payload(lowest_both_presence),
        "both_hands_present_ratio",
        "both_hand_presence",
    )
    print_video_ranking(
        "20 VIDEOS WITH HIGHEST BOTH-HAND PRESENCE",
        top_record_payload(highest_both_presence),
        "both_hands_present_ratio",
        "both_hand_presence",
    )
    print_video_ranking(
        "20 VIDEOS WITH LOWEST HAND MOVEMENT",
        top_record_payload(lowest_hand_movement),
        "hand_movement_mean",
        "hand_movement_mean",
    )
    print_video_ranking(
        "20 VIDEOS WITH HIGHEST HAND MOVEMENT",
        top_record_payload(highest_hand_movement),
        "hand_movement_mean",
        "hand_movement_mean",
    )
    print_video_ranking(
        "20 VIDEOS WITH HIGHEST ZERO-COORDINATE PERCENTAGE",
        top_record_payload(highest_zero_percent),
        "zero_coordinate_percentage",
        "zero_coordinate_percentage",
    )
    print_video_ranking(
        "20 VIDEOS WITH HIGHEST TEMPORAL JITTER",
        top_record_payload(highest_jitter),
        "temporal_jitter",
        "jitter_mean",
    )

    report: dict[str, Any] = {
        "dataset_summary": {
            "seed": SEED,
            "split": "train.csv",
            "rows_in_training_split": int(len(train_frame)),
            "videos_analyzed": len(valid_records),
            "invalid_files": len(invalid_records),
            "shape_invalid_files": sum(
                not record["shape_valid"] for record in all_records
            ),
            "dtype_invalid_files": sum(
                not record["dtype_valid"] for record in all_records
            ),
            "class_count": int(train_frame["class_name"].nunique()),
            "feature_dimension": FEATURE_DIMENSION,
            "landmarks_per_frame": LANDMARK_COUNT,
            "landmark_components": {
                "POSE": "landmarks 0:33; features 0:99",
                "LEFT_HAND": "landmarks 33:54; features 99:162",
                "RIGHT_HAND": "landmarks 54:75; features 162:225",
                "BOTH_HANDS": "landmarks 33:75; features 99:225",
            },
            "invalid_file_records": invalid_records,
        },
        "thresholds": {
            "nearly_static_definition": (
                "BOTH_HANDS frame-to-frame Euclidean movement at or below the "
                "empirical 5th percentile of all finite training transitions."
            ),
            "nearly_static_movement_threshold": static_threshold,
            "nearly_static_threshold_percentile": 5,
            "note": "Data-derived descriptive threshold; not a quality verdict.",
        },
        "overall_statistics": aggregate,
        "component_statistics": component_aggregate,
        "videos": all_records,
        "class_level_statistics": class_records,
        "classes_with_high_internal_variation": class_variability,
        "ranked_videos": {
            "lowest_both_hand_presence": top_record_payload(lowest_both_presence),
            "highest_both_hand_presence": top_record_payload(highest_both_presence),
            "lowest_hand_movement": top_record_payload(lowest_hand_movement),
            "highest_hand_movement": top_record_payload(highest_hand_movement),
            "highest_zero_coordinate_percentage": top_record_payload(
                highest_zero_percent
            ),
            "highest_temporal_jitter": top_record_payload(highest_jitter),
        },
        "interpretation": {},
    }

    left_presence = aggregate["left_hand_presence_ratio"]["mean"]
    right_presence = aggregate["right_hand_presence_ratio"]["mean"]
    both_presence = aggregate["both_hands_present_ratio"]["mean"]
    invalid_file_count = len(invalid_records)
    if (
        invalid_file_count == 0
        and aggregate["nan_inf_frame_percentage"]["mean"] == 0
        and aggregate["nan_inf_coordinate_percentage"]["mean"] == 0
    ):
        consistency = (
            "All loaded files passed shape and float32 dtype validation, and contained "
            "no NaN/Inf values; this supports structural and numeric consistency for "
            "the analyzed split."
        )
    else:
        consistency = (
            f"{invalid_file_count} file(s) failed loading, shape, or dtype validation; "
            "reported NaN/Inf rates describe analyzable files."
        )
    if left_presence is not None and right_presence is not None and both_presence is not None:
        missing_hands = (
            "By the specified all-zero rule, the measured average frames without "
            "left hand, right hand, or both hands are "
            f"{(1.0 - left_presence) * 100.0:.2f}%, "
            f"{(1.0 - right_presence) * 100.0:.2f}%, and "
            f"{(1.0 - both_presence) * 100.0:.2f}% respectively. "
            "These describe stored normalized coordinates under that exact rule, "
            "not an independent confirmation that a hand detector fired."
        )
    else:
        missing_hands = "Hand presence could not be summarized from valid files."
    quality_note = (
        "The ranked video lists show relative extremes in hand presence, movement, "
        "zero-coordinate percentage, and jitter within this training split. These "
        "rankings are not pass/fail criteria."
    )
    if (
        left_presence == 1.0
        and right_presence == 1.0
        and aggregate["zero_coordinate_percentage"]["mean"] is not None
        and aggregate["zero_coordinate_percentage"]["mean"] < 0.01
    ):
        representation_note = (
            "Although file shape, dtype, and finiteness are consistent, the all-zero "
            "presence rule reports every hand as present while almost no coordinates "
            "are exactly zero. Review how missing landmarks are represented after "
            "normalization before treating these presence ratios as detector coverage."
        )
    else:
        representation_note = (
            "File-integrity and presence results, together with the reported "
            "per-class/per-video movement, jitter, and zero-coordinate distributions, "
            "can be used to decide whether representation quality merits further "
            "investigation. They do not recommend an architecture."
        )
    report["interpretation"] = {
        "extraction_consistency": consistency,
        "hand_missingness": missing_hands,
        "unusually_poor_video_screen": quality_note,
        "representation_follow_up": representation_note,
        "caveat": (
            "Interpretations are descriptive and use only the training split. "
            "The threshold-based screening text is not a judgement that landmarks "
            "or the dataset are good or bad."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print("\nLANDMARK QUALITY SUMMARY")
    print(f"Videos analyzed: {len(valid_records)}")
    print(f"Invalid files: {len(invalid_records)}")
    print(f"Mean left-hand presence: {left_presence:.4f}" if left_presence is not None else "Mean left-hand presence: n/a")
    print(f"Mean right-hand presence: {right_presence:.4f}" if right_presence is not None else "Mean right-hand presence: n/a")
    print(f"Mean both-hand presence: {both_presence:.4f}" if both_presence is not None else "Mean both-hand presence: n/a")
    print(
        f"Mean hand movement: "
        f"{component_aggregate['BOTH_HANDS']['frame_to_frame_movement_per_video_mean']['mean']:.6f}"
        if component_aggregate["BOTH_HANDS"]["frame_to_frame_movement_per_video_mean"]["mean"] is not None
        else "Mean hand movement: n/a"
    )
    print(
        f"Mean pose movement: "
        f"{component_aggregate['POSE']['frame_to_frame_movement_per_video_mean']['mean']:.6f}"
        if component_aggregate["POSE"]["frame_to_frame_movement_per_video_mean"]["mean"] is not None
        else "Mean pose movement: n/a"
    )
    print(
        f"Mean zero-coordinate percentage: "
        f"{aggregate['zero_coordinate_percentage']['mean']:.4f}"
        if aggregate["zero_coordinate_percentage"]["mean"] is not None
        else "Mean zero-coordinate percentage: n/a"
    )
    print(
        "Mean jitter: "
        f"{aggregate['temporal_jitter_both_hands_all_transitions']['mean']:.6f}"
        if aggregate["temporal_jitter_both_hands_all_transitions"]["mean"] is not None
        else "Mean jitter: n/a"
    )
    print(
        f"Nearly-static transition threshold (empirical P5): "
        f"{static_threshold:.6f}" if static_threshold is not None
        else "Nearly-static transition threshold: n/a"
    )
    print("\nNEUTRAL INTERPRETATION")
    print(consistency)
    print(missing_hands)
    print(quality_note)
    print(representation_note)
    print(f"\nJSON report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
