"""Measure class-distance and nearest-neighbor statistics for ISL landmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "splits" / "val.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "class_separability_report.json"
FEATURE_DIMENSION = 225
RESAMPLED_FRAME_COUNT = 32
MOTION_FRAME_COUNT = RESAMPLED_FRAME_COUNT - 1
SEED = 42


def resolve_landmark_path(value: str) -> Path:
    """Resolve split paths against the repository root when they are relative."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_sentence_mapping(path: Path) -> dict[str, str]:
    """Load one consistent human-readable sentence for each manifest label."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest = pd.read_csv(path, keep_default_na=False)
    required_columns = {"label_id", "sentence"}
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    sentence_by_label: dict[str, str] = {}
    for row_number, row in manifest.iterrows():
        label = str(row["label_id"]).strip()
        sentence = str(row["sentence"]).strip()
        if not label or not sentence:
            raise ValueError(
                f"Manifest has an empty label_id or sentence at data row {row_number + 2}."
            )
        existing_sentence = sentence_by_label.setdefault(label, sentence)
        if existing_sentence != sentence:
            raise ValueError(
                f"Manifest has conflicting sentences for {label}: "
                f"{existing_sentence!r} and {sentence!r}."
            )
    if not sentence_by_label:
        raise ValueError(f"Manifest contains no class sentences: {path}")
    return sentence_by_label


def resample_sequence(sequence: np.ndarray, frame_count: int) -> np.ndarray:
    """Linearly resample frames over normalized time, preserving feature order."""
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected landmark shape (T, {FEATURE_DIMENSION}), got {sequence.shape}."
        )
    if sequence.shape[0] == 0:
        raise ValueError("Landmark sequence has no valid frames to resample.")

    sample_positions = np.linspace(
        0.0,
        float(sequence.shape[0] - 1),
        num=frame_count,
        dtype=np.float64,
    )
    left_indices = np.floor(sample_positions).astype(np.intp)
    right_indices = np.minimum(left_indices + 1, sequence.shape[0] - 1)
    interpolation_weights = (sample_positions - left_indices).astype(np.float32)[:, None]
    sampled = (
        sequence[left_indices] * (1.0 - interpolation_weights)
        + sequence[right_indices] * interpolation_weights
    )
    return sampled.astype(np.float32, copy=False)


def load_split_features(
    split_name: str,
    split_path: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], dict[str, Any], list[dict[str, str]]]:
    """Load split samples, drop invalid frames, and create position/motion vectors."""
    if not split_path.is_file():
        raise FileNotFoundError(f"{split_name} split not found: {split_path}")

    frame = pd.read_csv(split_path, keep_default_na=False)
    required_columns = {"landmark_path", "label_id"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{split_path} is missing required columns: {sorted(missing_columns)}"
        )

    position_features: list[np.ndarray] = []
    motion_features: list[np.ndarray] = []
    labels: list[str] = []
    sentences: list[str] = []
    issues: list[dict[str, str]] = []
    invalid_frames_removed = 0
    missing_sentences: set[str] = set()

    for row_index, row in frame.iterrows():
        label = str(row["label_id"]).strip()
        sentence = str(row.get("sentence", "")).strip()
        raw_path = str(row["landmark_path"]).strip()
        landmark_path = resolve_landmark_path(raw_path)
        try:
            if not label:
                raise ValueError("Split row has an empty label_id.")
            if not raw_path:
                raise ValueError("Split row has an empty landmark_path.")
            sequence = np.load(landmark_path, allow_pickle=False)
            if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIMENSION:
                raise ValueError(
                    f"Expected shape (T, {FEATURE_DIMENSION}), got {sequence.shape}."
                )
            if sequence.shape[0] == 0:
                raise ValueError("Landmark sequence has zero frames.")
            try:
                sequence = np.asarray(sequence, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Landmark values are not numeric: {exc}") from exc

            valid_frame_mask = np.isfinite(sequence).all(axis=1)
            invalid_count = int((~valid_frame_mask).sum())
            if invalid_count:
                invalid_frames_removed += invalid_count
                print(
                    f"Warning: {split_name} row {row_index + 2}, {landmark_path}: "
                    f"removed {invalid_count} frame(s) containing NaN/Inf."
                )
                sequence = sequence[valid_frame_mask]
            if sequence.shape[0] == 0:
                raise ValueError("No finite frames remain after removing invalid frames.")

            resampled = resample_sequence(sequence, RESAMPLED_FRAME_COUNT)
            delta = np.diff(resampled, axis=0)
            if delta.shape != (MOTION_FRAME_COUNT, FEATURE_DIMENSION):
                raise ValueError(
                    f"Expected delta shape ({MOTION_FRAME_COUNT}, {FEATURE_DIMENSION}), "
                    f"got {delta.shape}."
                )
            position_features.append(resampled.reshape(-1))
            motion_features.append(delta.reshape(-1))
            labels.append(label)
            sentences.append(sentence)
            if not sentence:
                missing_sentences.add(label)
        except (OSError, ValueError, TypeError, EOFError) as exc:
            issue = {
                "split": split_name,
                "row_number": str(row_index + 2),
                "label_id": label,
                "landmark_path": str(landmark_path),
                "error": str(exc),
            }
            issues.append(issue)
            print(
                f"Warning: skipping {split_name} row {row_index + 2} "
                f"({landmark_path}): {exc}"
            )

    summary = {
        "rows_in_split": int(len(frame)),
        "samples_used": len(labels),
        "classes_used": len(set(labels)),
        "invalid_frames_removed": invalid_frames_removed,
        "samples_skipped": len(issues),
        "missing_split_sentences": sorted(missing_sentences),
    }
    return position_features, motion_features, labels, sentences, summary, issues


def as_feature_matrix(features: list[np.ndarray], description: str) -> np.ndarray:
    """Stack validated per-sample features into a finite two-dimensional matrix."""
    if not features:
        raise ValueError(f"No valid {description} samples are available.")
    matrix = np.stack(features).astype(np.float64, copy=False)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{description} feature matrix is not a finite 2D matrix.")
    return matrix


def distance_statistics(distances: np.ndarray) -> dict[str, float | int | None]:
    """Summarize a one-dimensional set of Euclidean distances."""
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "pair_count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "pair_count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def train_pairwise_distance_statistics(
    train_features: np.ndarray,
    train_labels: list[str],
) -> tuple[dict[str, float | int | None], dict[str, float | int | None]]:
    """Compute upper-triangle within- and between-class training distances."""
    distances = euclidean_distances(train_features, train_features)
    upper_rows, upper_cols = np.triu_indices(len(train_labels), k=1)
    same_class = np.equal(
        np.asarray(train_labels, dtype=object)[upper_rows],
        np.asarray(train_labels, dtype=object)[upper_cols],
    )
    pair_distances = distances[upper_rows, upper_cols]
    return (
        distance_statistics(pair_distances[same_class]),
        distance_statistics(pair_distances[~same_class]),
    )


def nearest_neighbor_analysis(
    train_features: np.ndarray,
    train_labels: list[str],
    validation_features: np.ndarray,
    validation_labels: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    """Find the closest training sample for every validation sample."""
    distances = euclidean_distances(validation_features, train_features)
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(nearest_indices)), nearest_indices]
    same_class = np.asarray(
        [
            validation_labels[index] == train_labels[train_index]
            for index, train_index in enumerate(nearest_indices)
        ],
        dtype=bool,
    )
    sample_count = len(validation_labels)
    stats: dict[str, Any] = {
        "samples": sample_count,
        "same_class_count": int(same_class.sum()),
        "different_class_count": int(sample_count - same_class.sum()),
        "same_class_percentage": float(100.0 * same_class.mean()) if sample_count else None,
        "different_class_percentage": float(100.0 * (~same_class).mean()) if sample_count else None,
        "mean_nearest_distance": float(np.mean(nearest_distances)) if sample_count else None,
        "median_nearest_distance": float(np.median(nearest_distances)) if sample_count else None,
    }
    return stats, np.column_stack((nearest_indices, nearest_distances))


def closest_class_pairs(
    train_features: np.ndarray,
    train_labels: list[str],
    validation_features: np.ndarray,
    validation_labels: list[str],
    sentence_by_label: dict[str, str],
) -> list[dict[str, Any]]:
    """Find each validation class's closest different training class."""
    training_labels = np.asarray(train_labels, dtype=object)
    validation_label_array = np.asarray(validation_labels, dtype=object)
    training_classes = sorted(set(train_labels))
    output: list[dict[str, Any]] = []

    for class_id in sorted(set(validation_labels)):
        validation_class_features = validation_features[
            validation_label_array == class_id
        ]
        candidate_distances: list[tuple[str, float]] = []
        for other_class in training_classes:
            if other_class == class_id:
                continue
            other_class_features = train_features[training_labels == other_class]
            if len(other_class_features) == 0:
                continue
            distances = euclidean_distances(
                validation_class_features,
                other_class_features,
            )
            candidate_distances.append((other_class, float(distances.mean())))

        if not candidate_distances:
            continue
        closest_other_class, average_distance = min(
            candidate_distances,
            key=lambda item: (item[1], item[0]),
        )
        output.append(
            {
                "class": class_id,
                "sentence": sentence_by_label.get(class_id),
                "closest_other_class": closest_other_class,
                "closest_other_sentence": sentence_by_label.get(closest_other_class),
                "average_distance_to_closest_other_class": average_distance,
            }
        )
    return sorted(
        output,
        key=lambda item: (
            item["average_distance_to_closest_other_class"],
            item["class"],
            item["closest_other_class"],
        ),
    )


def build_validation_pairs(
    nearest: np.ndarray,
    train_labels: list[str],
    validation_labels: list[str],
    validation_sentences: list[str],
) -> list[dict[str, Any]]:
    """Create validation-to-nearest-training records ordered by distance."""
    pairs: list[dict[str, Any]] = []
    for validation_index, (raw_train_index, raw_distance) in enumerate(nearest):
        train_index = int(raw_train_index)
        validation_label = validation_labels[validation_index]
        nearest_train_label = train_labels[train_index]
        pairs.append(
            {
                "validation_class": validation_label,
                "validation_sentence": validation_sentences[validation_index] or None,
                "nearest_train_class": nearest_train_label,
                "nearest_train_sentence": None,
                "distance": float(raw_distance),
                "same_class": validation_label == nearest_train_label,
                "_train_index": train_index,
            }
        )
    pairs.sort(key=lambda item: (item["distance"], item["validation_class"], item["_train_index"]))
    for pair in pairs:
        pair.pop("_train_index")
    return pairs


def add_manifest_sentences_to_pairs(
    pairs: list[dict[str, Any]],
    sentence_by_label: dict[str, str],
) -> None:
    """Resolve both class sentences through manifest mappings."""
    for pair in pairs:
        pair["validation_sentence"] = sentence_by_label.get(
            pair["validation_class"]
        )
        pair["nearest_train_sentence"] = sentence_by_label.get(
            pair["nearest_train_class"]
        )


def print_distance_stats(title: str, stats: dict[str, float | int | None]) -> None:
    """Print the pair count and standard descriptive distance statistics."""
    print(f"\n{title}")
    print(f"Number of pairs: {stats['pair_count']}")
    for key, display in (
        ("mean", "Mean"),
        ("median", "Median"),
        ("minimum", "Minimum"),
        ("maximum", "Maximum"),
    ):
        value = stats[key]
        print(f"{display}: {value:.6f}" if value is not None else f"{display}: n/a")


def main() -> int:
    """Run separability analyses and save a JSON report."""
    np.random.seed(SEED)
    sentence_by_label = load_sentence_mapping(MANIFEST_PATH)
    train_position, train_motion, train_labels, _, train_summary, train_issues = (
        load_split_features("TRAIN", TRAIN_PATH)
    )
    val_position, val_motion, val_labels, val_sentences, val_summary, val_issues = (
        load_split_features("VALIDATION", VALIDATION_PATH)
    )
    if not train_labels:
        raise ValueError("No valid training samples remain for analysis.")
    if not val_labels:
        raise ValueError("No valid validation samples remain for analysis.")

    train_position_matrix = as_feature_matrix(train_position, "training position")
    val_position_matrix = as_feature_matrix(val_position, "validation position")
    train_motion_matrix = as_feature_matrix(train_motion, "training motion")
    val_motion_matrix = as_feature_matrix(val_motion, "validation motion")

    position_scaler = StandardScaler()
    scaled_train_position = position_scaler.fit_transform(train_position_matrix)
    scaled_val_position = position_scaler.transform(val_position_matrix)
    motion_scaler = StandardScaler()
    scaled_train_motion = motion_scaler.fit_transform(train_motion_matrix)
    scaled_val_motion = motion_scaler.transform(val_motion_matrix)

    within_stats, between_stats = train_pairwise_distance_statistics(
        scaled_train_position,
        train_labels,
    )
    position_nn_stats, position_nearest = nearest_neighbor_analysis(
        scaled_train_position,
        train_labels,
        scaled_val_position,
        val_labels,
    )
    motion_nn_stats, _ = nearest_neighbor_analysis(
        scaled_train_motion,
        train_labels,
        scaled_val_motion,
        val_labels,
    )
    class_pairs = closest_class_pairs(
        scaled_train_position,
        train_labels,
        scaled_val_position,
        val_labels,
        sentence_by_label,
    )
    validation_pairs = build_validation_pairs(
        position_nearest,
        train_labels,
        val_labels,
        val_sentences,
    )
    add_manifest_sentences_to_pairs(validation_pairs, sentence_by_label)
    closest_validation_pairs = validation_pairs[:20]
    closest_cross_class_pairs = class_pairs[:20]

    mean_within = within_stats["mean"]
    mean_between = between_stats["mean"]
    if mean_within is None or mean_between is None:
        separability_ratio = None
        distance_margin = None
    else:
        separability_ratio = (
            float(mean_between / mean_within) if mean_within != 0 else None
        )
        distance_margin = float(mean_between - mean_within)

    dataset_summary = {
        "seed": SEED,
        "landmark_dimension": FEATURE_DIMENSION,
        "resampled_frames": RESAMPLED_FRAME_COUNT,
        "position_feature_dimension": RESAMPLED_FRAME_COUNT * FEATURE_DIMENSION,
        "motion_delta_frames": MOTION_FRAME_COUNT,
        "motion_feature_dimension": MOTION_FRAME_COUNT * FEATURE_DIMENSION,
        "train": train_summary,
        "validation": val_summary,
        "class_count_in_manifest": len(sentence_by_label),
        "load_issues": train_issues + val_issues,
    }
    report: dict[str, Any] = {
        "dataset_summary": dataset_summary,
        "position_distance_statistics": {
            "within_class": within_stats,
            "between_class": between_stats,
            "separability_ratio": separability_ratio,
            "distance_margin": distance_margin,
            "interpretation_note": (
                "These are diagnostic Euclidean distance statistics, not accuracy."
            ),
        },
        "position_nearest_neighbor_statistics": position_nn_stats,
        "closest_validation_pairs": closest_validation_pairs,
        "closest_cross_class_class_pairs": closest_cross_class_pairs,
        "motion_delta_nearest_neighbor_statistics": motion_nn_stats,
    }

    print("============================================================")
    print("ISL BRIDGE FEATURE SEPARABILITY DIAGNOSTIC")
    print("============================================================")
    print("\nDATASET SUMMARY")
    print(f"Seed: {SEED}")
    print(f"Manifest classes: {len(sentence_by_label)}")
    print(
        f"Train samples: {train_summary['samples_used']} used / "
        f"{train_summary['rows_in_split']} rows"
    )
    print(
        f"Validation samples: {val_summary['samples_used']} used / "
        f"{val_summary['rows_in_split']} rows"
    )
    print(
        "Invalid frames removed: "
        f"train={train_summary['invalid_frames_removed']}, "
        f"validation={val_summary['invalid_frames_removed']}"
    )
    print(
        "Samples skipped due to missing/corrupt data: "
        f"train={train_summary['samples_skipped']}, "
        f"validation={val_summary['samples_skipped']}"
    )
    print(
        f"Position features: {RESAMPLED_FRAME_COUNT} x {FEATURE_DIMENSION} = "
        f"{RESAMPLED_FRAME_COUNT * FEATURE_DIMENSION}"
    )
    print(
        f"Motion features: {MOTION_FRAME_COUNT} x {FEATURE_DIMENSION} = "
        f"{MOTION_FRAME_COUNT * FEATURE_DIMENSION}"
    )
    print("StandardScaler: fitted on training samples only for each representation.")

    print("\nPOSITION FEATURE SEPARABILITY")
    print("Euclidean distances after training-only StandardScaler; not accuracy.")
    print_distance_stats("WITHIN-CLASS DISTANCES", within_stats)
    print_distance_stats("BETWEEN-CLASS DISTANCES", between_stats)

    print("\nSEPARABILITY SUMMARY")
    print(
        f"Separability ratio (mean between / mean within): "
        f"{separability_ratio:.6f}" if separability_ratio is not None
        else "Separability ratio: n/a"
    )
    print(
        f"Distance margin (mean between - mean within): {distance_margin:.6f}"
        if distance_margin is not None
        else "Distance margin: n/a"
    )
    if mean_within is not None and mean_between is not None:
        if mean_between > mean_within:
            print(
                "Between-class distances are larger than within-class distances, "
                "indicating some global class separation."
            )
        else:
            print(
                "Between-class distances are not larger than within-class distances, "
                "indicating weak global separation."
            )
    print("These are diagnostic distance statistics, not accuracy.")

    print("\nVALIDATION NEAREST-NEIGHBOR ANALYSIS")
    print(f"Validation samples: {position_nn_stats['samples']}")
    print(
        f"Nearest-neighbor same-class percentage: "
        f"{position_nn_stats['same_class_percentage']:.2f}%"
    )
    print(
        f"Nearest-neighbor different-class percentage: "
        f"{position_nn_stats['different_class_percentage']:.2f}%"
    )
    print(f"Mean nearest distance: {position_nn_stats['mean_nearest_distance']:.6f}")
    print(f"Median nearest distance: {position_nn_stats['median_nearest_distance']:.6f}")

    print("\n20 CLOSEST VALIDATION -> TRAIN PAIRS")
    for rank, pair in enumerate(closest_validation_pairs, start=1):
        validation_sentence = pair["validation_sentence"] or "<sentence unavailable>"
        train_sentence = pair["nearest_train_sentence"] or "<sentence unavailable>"
        print(
            f"{rank:2d}. validation={pair['validation_class']} "
            f'("{validation_sentence}") -> train={pair["nearest_train_class"]} '
            f'("{train_sentence}") | distance={pair["distance"]:.6f} | '
            f"same_class={str(pair['same_class']).lower()}"
        )

    print("\nCLASS-LEVEL CROSS-CLASS SIMILARITY")
    print("20 class pairs with the smallest average cross-class distance:")
    for rank, pair in enumerate(closest_cross_class_pairs, start=1):
        source_sentence = pair["sentence"] or "<sentence unavailable>"
        other_sentence = pair["closest_other_sentence"] or "<sentence unavailable>"
        print(
            f"{rank:2d}. {pair['class']} ({source_sentence}) -> "
            f"{pair['closest_other_class']} ({other_sentence}) | "
            f"average_distance={pair['average_distance_to_closest_other_class']:.6f}"
        )

    print("\nMOTION/DELTA FEATURE ANALYSIS")
    print(
        f"Nearest-neighbor same-class percentage: "
        f"{motion_nn_stats['same_class_percentage']:.2f}%"
    )
    print(
        f"Nearest-neighbor different-class percentage: "
        f"{motion_nn_stats['different_class_percentage']:.2f}%"
    )
    print(f"Mean nearest distance: {motion_nn_stats['mean_nearest_distance']:.6f}")
    print(f"Median nearest distance: {motion_nn_stats['median_nearest_distance']:.6f}")
    print(
        f"Position same-class percentage: "
        f"{position_nn_stats['same_class_percentage']:.2f}%"
    )

    print("\nFINAL DIAGNOSTIC INTERPRETATION")
    if mean_within is not None and mean_between is not None:
        if mean_between > mean_within:
            print(
                "Between-class distances are larger than within-class distances, "
                "indicating some global class separation."
            )
        else:
            print(
                "Between-class distances are not larger than within-class distances, "
                "indicating weak global separation."
            )
    print(
        f"Measured validation nearest-neighbor same-class percentages: "
        f"position={position_nn_stats['same_class_percentage']:.2f}%, "
        f"motion/delta={motion_nn_stats['same_class_percentage']:.2f}%."
    )
    print(
        "This diagnostic does not measure final classification accuracy and does not "
        "determine the next architecture by itself."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nJSON report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(
        "Run command: .\\.venv310\\Scripts\\python.exe scripts\\class_separability.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
