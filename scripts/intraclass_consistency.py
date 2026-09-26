"""Analyze within-class consistency and cross-class distances with NumPy DTW."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "intraclass_consistency_report.json"

FEATURE_DIMENSION = 225
RESAMPLED_FRAMES = 32
DTW_WINDOW = 8
SEED = 42
REPRESENTATIONS: dict[str, tuple[slice, ...]] = {
    "BOTH_HANDS": (slice(99, 225),),
    "POSE_PLUS_BOTH_HANDS": (slice(0, 99), slice(99, 225)),
}


def resolve_path(value: str) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_manifest_sentences(path: Path) -> dict[str, str]:
    """Load a consistent human-readable sentence for every label."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest = pd.read_csv(path, keep_default_na=False)
    required = {"label_id", "sentence"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    sentences: dict[str, str] = {}
    for row_index, row in manifest.iterrows():
        label = str(row["label_id"]).strip()
        sentence = str(row["sentence"]).strip()
        if not label or not sentence:
            raise ValueError(
                f"Manifest has an empty label_id or sentence at row {row_index + 2}."
            )
        previous = sentences.setdefault(label, sentence)
        if previous != sentence:
            raise ValueError(f"Manifest contains conflicting sentences for {label}.")
    return sentences


def resample_sequence(sequence: np.ndarray) -> np.ndarray:
    """Deterministically linearly resample frames to the common temporal grid."""
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected landmark shape (T, {FEATURE_DIMENSION}), got {sequence.shape}."
        )
    if sequence.shape[0] == 0:
        raise ValueError("Landmark sequence contains no valid frames.")

    positions = np.linspace(
        0.0,
        float(sequence.shape[0] - 1),
        RESAMPLED_FRAMES,
        dtype=np.float64,
    )
    left = np.floor(positions).astype(np.intp)
    right = np.minimum(left + 1, sequence.shape[0] - 1)
    alpha = (positions - left).astype(np.float32)[:, None]
    return (
        sequence[left] * (1.0 - alpha) + sequence[right] * alpha
    ).astype(np.float32, copy=False)


def load_training_sequences(
    split_path: Path,
) -> tuple[list[np.ndarray], list[str], list[str], dict[str, int], list[dict[str, str]]]:
    """Load every usable training landmark sequence and report skipped rows."""
    if not split_path.is_file():
        raise FileNotFoundError(f"Training split not found: {split_path}")
    frame = pd.read_csv(split_path, keep_default_na=False)
    required = {"landmark_path", "label_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split_path} is missing required columns: {sorted(missing)}")

    sequences: list[np.ndarray] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    issues: list[dict[str, str]] = []
    invalid_frames_removed = 0

    for row_index, row in frame.iterrows():
        raw_path = str(row["landmark_path"]).strip()
        label = str(row["label_id"]).strip()
        landmark_path = resolve_path(raw_path)
        try:
            if not raw_path or not label:
                raise ValueError("Split row has an empty landmark_path or label_id.")
            values = np.load(landmark_path, allow_pickle=False)
            if values.ndim != 2 or values.shape[1] != FEATURE_DIMENSION:
                raise ValueError(
                    f"Expected shape (T, {FEATURE_DIMENSION}), got {values.shape}."
                )
            if values.shape[0] == 0:
                raise ValueError("Landmark sequence contains zero frames.")
            try:
                values = np.asarray(values, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Landmark values are not numeric: {exc}") from exc
            valid_frames = np.isfinite(values).all(axis=1)
            removed = int((~valid_frames).sum())
            if removed:
                invalid_frames_removed += removed
                print(
                    f"Warning: train row {row_index + 2} ({landmark_path}): "
                    f"removing {removed} frame(s) containing NaN/Inf."
                )
                values = values[valid_frames]
            if values.shape[0] == 0:
                raise ValueError("No finite frames remain after invalid frames were removed.")

            sequences.append(resample_sequence(values))
            labels.append(label)
            sample_ids.append(f"train_row_{row_index + 2}:{landmark_path.name}")
        except (OSError, ValueError, TypeError, EOFError) as exc:
            issue = {
                "row_number": str(row_index + 2),
                "label_id": label,
                "landmark_path": str(landmark_path),
                "error": str(exc),
            }
            issues.append(issue)
            print(f"Warning: skipping train row {row_index + 2}: {exc}")

    if not sequences:
        raise ValueError(f"No valid training landmark files found in {split_path}.")
    return (
        sequences,
        labels,
        sample_ids,
        {
            "rows_in_split": int(len(frame)),
            "samples_used": len(sequences),
            "classes_used": len(set(labels)),
            "invalid_frames_removed": invalid_frames_removed,
            "samples_skipped": len(issues),
        },
        issues,
    )


def select_representation(
    sequences: list[np.ndarray],
    component_slices: tuple[slice, ...],
) -> np.ndarray:
    """Select a feature representation into (samples, frames, dimensions)."""
    represented = [
        np.concatenate([sequence[:, part] for part in component_slices], axis=1)
        for sequence in sequences
    ]
    values = np.stack(represented).astype(np.float64, copy=False)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Selected representation must be a finite (N,T,D) array.")
    return values


def dtw_distances_to_candidates(
    query: np.ndarray,
    candidates: np.ndarray,
    window: int = DTW_WINDOW,
) -> np.ndarray:
    """Compute cumulative Euclidean-cost DTW from one query to candidates."""
    if query.ndim != 2 or candidates.ndim != 3:
        raise ValueError("DTW expects query (T,D) and candidates (N,T,D).")
    if candidates.shape[1:] != query.shape:
        raise ValueError(
            f"DTW sequence shapes must match; got {query.shape} and "
            f"{candidates.shape[1:]}."
        )

    time_steps = query.shape[0]
    candidate_count = candidates.shape[0]
    band = max(window, abs(time_steps - candidates.shape[1]))
    previous = np.full((time_steps + 1, candidate_count), np.inf, dtype=np.float64)
    previous[0, :] = 0.0

    for query_index in range(1, time_steps + 1):
        current = np.full((time_steps + 1, candidate_count), np.inf, dtype=np.float64)
        start = max(1, query_index - band)
        end = min(time_steps, query_index + band)
        for candidate_index in range(start, end + 1):
            delta = candidates[:, candidate_index - 1, :] - query[query_index - 1]
            frame_cost = np.sqrt(np.einsum("nd,nd->n", delta, delta))
            best_previous = np.minimum(
                previous[candidate_index],
                np.minimum(
                    current[candidate_index - 1],
                    previous[candidate_index - 1],
                ),
            )
            current[candidate_index] = frame_cost + best_previous
        previous = current
    return previous[time_steps]


def pairwise_dtw_matrix(sequences: np.ndarray, representation: str) -> np.ndarray:
    """Compute a symmetric full pairwise DTW matrix on CPU."""
    sample_count = len(sequences)
    distances = np.empty((sample_count, sample_count), dtype=np.float64)
    for sample_index, query in enumerate(sequences):
        distances[sample_index] = dtw_distances_to_candidates(query, sequences)
        distances[sample_index, sample_index] = 0.0
        if (sample_index + 1) % 25 == 0 or sample_index + 1 == sample_count:
            print(
                f"  {representation} DTW progress: "
                f"{sample_index + 1}/{sample_count} samples"
            )
    # DTW can be slightly asymmetric numerically; pair statistics use one
    # deterministic symmetric value for each unordered pair.
    return (distances + distances.T) / 2.0


def pair_distance_summary(distances: np.ndarray) -> dict[str, int | float | None]:
    """Return count and descriptive statistics for distance values."""
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "number_of_pairs": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "number_of_pairs": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def analyze_distance_matrix(
    distances: np.ndarray,
    labels: list[str],
    sample_ids: list[str],
    sentence_by_label: dict[str, str],
) -> dict[str, Any]:
    """Calculate within/between statistics and class/sample confusion records."""
    label_array = np.asarray(labels, dtype=object)
    classes = sorted(set(labels))
    within_by_class: dict[str, dict[str, Any]] = {}
    within_all: list[np.ndarray] = []
    within_pair_records: list[dict[str, Any]] = []
    excluded_classes: list[str] = []
    class_indices = {
        class_id: np.flatnonzero(label_array == class_id)
        for class_id in classes
    }

    for class_id in classes:
        indices = class_indices[class_id]
        if len(indices) < 2:
            excluded_classes.append(class_id)
            continue
        pair_rows, pair_cols = np.triu_indices(len(indices), k=1)
        class_distances = distances[indices[pair_rows], indices[pair_cols]]
        within_all.append(class_distances)
        stats = pair_distance_summary(class_distances)
        within_by_class[class_id] = {
            "number_of_samples": int(len(indices)),
            **stats,
        }
        within_pair_records.extend(
            {
                "class": class_id,
                "sample_a": sample_ids[int(indices[first_position])],
                "sample_b": sample_ids[int(indices[second_position])],
                "distance": float(distances[indices[first_position], indices[second_position]]),
            }
            for first_position, second_position in zip(pair_rows, pair_cols)
        )

    within_values = np.concatenate(within_all) if within_all else np.array([])
    within_stats = pair_distance_summary(within_values)

    upper_rows, upper_cols = np.triu_indices(len(labels), k=1)
    is_between = label_array[upper_rows] != label_array[upper_cols]
    between_values = distances[upper_rows[is_between], upper_cols[is_between]]
    between_stats = pair_distance_summary(between_values)

    class_relationships: list[dict[str, Any]] = []
    for class_id in classes:
        class_sample_indices = class_indices[class_id]
        if len(class_sample_indices) < 2:
            continue
        other_class_distances: list[tuple[str, float]] = []
        for other_class in classes:
            if other_class == class_id:
                continue
            other_indices = class_indices[other_class]
            if len(other_indices) == 0:
                continue
            average_distance = float(
                distances[np.ix_(class_sample_indices, other_indices)].mean()
            )
            other_class_distances.append((other_class, average_distance))
        if not other_class_distances:
            continue
        closest_other_class, closest_distance = min(
            other_class_distances,
            key=lambda item: (item[1], item[0]),
        )
        class_relationships.append(
            {
                "class": class_id,
                "sentence": sentence_by_label.get(class_id),
                "number_of_training_samples": int(len(class_sample_indices)),
                "within_class_mean_distance": within_by_class[class_id]["mean"],
                "closest_other_class": closest_other_class,
                "closest_other_sentence": sentence_by_label.get(closest_other_class),
                "distance_to_closest_other_class": closest_distance,
            }
        )
    class_relationships.sort(
        key=lambda item: (
            item["distance_to_closest_other_class"],
            item["class"],
            item["closest_other_class"],
        )
    )

    sample_confusions: list[dict[str, Any]] = []
    for sample_index, class_id in enumerate(labels):
        other_indices = np.flatnonzero(label_array != class_id)
        if len(other_indices) == 0:
            continue
        other_distances = distances[sample_index, other_indices]
        nearest_position = int(np.argmin(other_distances))
        nearest_index = int(other_indices[nearest_position])
        sample_confusions.append(
            {
                "sample_class": class_id,
                "sample_sentence": sentence_by_label.get(class_id),
                "sample_id": sample_ids[sample_index],
                "nearest_other_class": labels[nearest_index],
                "nearest_other_sentence": sentence_by_label.get(labels[nearest_index]),
                "nearest_other_sample_id": sample_ids[nearest_index],
                "distance": float(distances[sample_index, nearest_index]),
            }
        )
    sample_confusions.sort(
        key=lambda item: (
            item["distance"],
            item["sample_class"],
            item["sample_id"],
        )
    )

    between_within_ratio = None
    if within_stats["mean"] not in (None, 0.0) and between_stats["mean"] is not None:
        between_within_ratio = float(between_stats["mean"] / within_stats["mean"])

    return {
        "within_class": {
            "per_class": within_by_class,
            "overall": within_stats,
            "pair_distances": within_pair_records,
            "classes_excluded_fewer_than_two_samples": excluded_classes,
            "excluded_class_count": len(excluded_classes),
        },
        "between_class": between_stats,
        "between_within_ratio": between_within_ratio,
        "class_relationships_sorted_by_closest_other_distance": class_relationships,
        "most_consistent_classes": sorted(
            class_relationships,
            key=lambda item: (
                item["within_class_mean_distance"],
                item["class"],
            ),
        )[:20],
        "most_variable_classes": sorted(
            class_relationships,
            key=lambda item: (
                -item["within_class_mean_distance"],
                item["class"],
            ),
        )[:20],
        "closest_cross_class_sample_pairs": sample_confusions[:20],
    }


def geometric_separation_label(ratio: float | None) -> str:
    """Map a measured between/within ratio to stated descriptive bands."""
    if ratio is None:
        return "not available"
    if ratio >= 1.5:
        return "strong"
    if ratio >= 1.1:
        return "moderate"
    return "weak"


def print_class_rows(
    title: str,
    rows: list[dict[str, Any]],
    sentence_field: str,
    value_field: str,
    value_label: str,
) -> None:
    """Print a ranked list of consistency or variability."""
    print(f"\n{title}")
    for rank, row in enumerate(rows, start=1):
        sentence = row.get(sentence_field) or "<sentence unavailable>"
        print(
            f"{rank:2d}. {row['class']} | {sentence} | "
            f"samples={row['number_of_training_samples']} | "
            f"{value_label}={row[value_field]:.6f}"
        )


def print_problem_relationships(rows: list[dict[str, Any]]) -> None:
    """Print the closest average cross-class relationships."""
    print("\n20 MOST PROBLEMATIC CLASS RELATIONSHIPS (smallest average cross-class distance)")
    for rank, row in enumerate(rows[:20], start=1):
        sentence = row.get("sentence") or "<sentence unavailable>"
        other_sentence = row.get("closest_other_sentence") or "<sentence unavailable>"
        print(
            f"{rank:2d}. {row['class']} ({sentence}) -> "
            f"{row['closest_other_class']} ({other_sentence}) | "
            f"samples={row['number_of_training_samples']} | "
            f"within_mean={row['within_class_mean_distance']:.6f} | "
            f"closest_other_distance={row['distance_to_closest_other_class']:.6f}"
        )


def print_sample_confusions(rows: list[dict[str, Any]]) -> None:
    """Print the closest sample-to-other-class pairs."""
    print("\n20 CLOSEST CROSS-CLASS SAMPLE PAIRS")
    for rank, row in enumerate(rows[:20], start=1):
        sentence = row.get("sample_sentence") or "<sentence unavailable>"
        other_sentence = row.get("nearest_other_sentence") or "<sentence unavailable>"
        print(
            f"{rank:2d}. {row['sample_class']} ({sentence}) -> "
            f"{row['nearest_other_class']} ({other_sentence}) | "
            f"{row['sample_id']} -> {row['nearest_other_sample_id']} | "
            f"distance={row['distance']:.6f}"
        )


def main() -> int:
    """Run train-only intraclass consistency diagnostics and save JSON."""
    np.random.seed(SEED)
    sentence_by_label = load_manifest_sentences(MANIFEST_PATH)
    sequences, labels, sample_ids, load_summary, load_issues = load_training_sequences(
        TRAIN_PATH
    )
    class_counts = pd.Series(labels, dtype="string").value_counts()
    classes_with_one_sample = sorted(
        class_id for class_id, count in class_counts.items() if int(count) < 2
    )

    print("============================================================")
    print("ISL BRIDGE INTRACLASS CONSISTENCY DIAGNOSTIC")
    print("============================================================")
    print("\nDATASET SUMMARY")
    print(f"Split: {TRAIN_PATH.relative_to(PROJECT_ROOT)} only")
    print(f"Samples loaded: {load_summary['samples_used']}")
    print(f"Classes loaded: {load_summary['classes_used']}")
    print(f"Manifest classes: {len(sentence_by_label)}")
    print(f"Resampled frames: {RESAMPLED_FRAMES}")
    print(f"Landmark dimension: {FEATURE_DIMENSION}")
    print(f"Invalid frames removed: {load_summary['invalid_frames_removed']}")
    print(f"Missing/corrupt samples skipped: {load_summary['samples_skipped']}")
    print(
        "Classes excluded from within-class statistics because they have fewer "
        f"than 2 samples: {len(classes_with_one_sample)}"
    )
    print("No validation or test samples are used.")
    print(f"DTW: Euclidean frame cost, Sakoe-Chiba window={DTW_WINDOW}, CPU only.")

    results: dict[str, Any] = {
        "dataset": {
            "seed": SEED,
            "split": "train.csv",
            "samples_in_split": load_summary["rows_in_split"],
            "samples_loaded": load_summary["samples_used"],
            "classes_loaded": load_summary["classes_used"],
            "resampled_frames": RESAMPLED_FRAMES,
            "landmark_dimension": FEATURE_DIMENSION,
            "invalid_frames_removed": load_summary["invalid_frames_removed"],
            "samples_skipped": load_summary["samples_skipped"],
            "load_issues": load_issues,
            "classes_excluded_fewer_than_two_training_samples": classes_with_one_sample,
            "excluded_class_count": len(classes_with_one_sample),
            "dtw": {
                "frame_cost": "Euclidean",
                "sakoe_chiba_window": DTW_WINDOW,
                "distance": "cumulative path cost",
            },
        },
        "representations": {},
    }

    for representation, component_slices in REPRESENTATIONS.items():
        print("\n" + "=" * 64)
        print(f"REPRESENTATION: {representation}")
        represented = select_representation(sequences, component_slices)
        print(f"Features per frame: {represented.shape[2]}")
        distance_matrix = pairwise_dtw_matrix(represented, representation)
        analysis = analyze_distance_matrix(
            distance_matrix,
            labels,
            sample_ids,
            sentence_by_label,
        )
        results["representations"][representation] = analysis

        within = analysis["within_class"]["overall"]
        between = analysis["between_class"]
        print("\nWITHIN-CLASS STATISTICS")
        print(
            f"Pairs: {within['number_of_pairs']} | mean={within['mean']:.6f} | "
            f"median={within['median']:.6f} | min={within['minimum']:.6f} | "
            f"max={within['maximum']:.6f}"
        )
        print("\nBETWEEN-CLASS STATISTICS")
        print(
            f"Pairs: {between['number_of_pairs']} | mean={between['mean']:.6f} | "
            f"median={between['median']:.6f} | min={between['minimum']:.6f} | "
            f"max={between['maximum']:.6f}"
        )
        ratio = analysis["between_within_ratio"]
        print(
            f"Between/within ratio: {ratio:.6f}"
            if ratio is not None
            else "Between/within ratio: n/a"
        )
        print_problem_relationships(
            analysis["class_relationships_sorted_by_closest_other_distance"]
        )
        print_class_rows(
            "20 MOST CONSISTENT CLASSES (lowest within-class mean distance)",
            analysis["most_consistent_classes"],
            "sentence",
            "within_class_mean_distance",
            "mean_within",
        )
        print_class_rows(
            "20 MOST VARIABLE CLASSES (highest within-class mean distance)",
            analysis["most_variable_classes"],
            "sentence",
            "within_class_mean_distance",
            "mean_within",
        )
        print_sample_confusions(analysis["closest_cross_class_sample_pairs"])

    both_hands = results["representations"]["BOTH_HANDS"]
    full_body = results["representations"]["POSE_PLUS_BOTH_HANDS"]
    both_ratio = both_hands["between_within_ratio"]
    full_ratio = full_body["between_within_ratio"]
    print("\n" + "=" * 64)
    print("HANDS VS FULL BODY COMPARISON")
    print("=" * 64)
    print(
        "Representation | Mean within-class distance | "
        "Mean between-class distance | Between/within ratio"
    )
    for name, analysis in (
        ("BOTH_HANDS", both_hands),
        ("POSE_PLUS_BOTH_HANDS", full_body),
    ):
        within_mean = analysis["within_class"]["overall"]["mean"]
        between_mean = analysis["between_class"]["mean"]
        ratio = analysis["between_within_ratio"]
        print(
            f"{name:<24} | {within_mean:>25.6f} | "
            f"{between_mean:>26.6f} | {ratio:>19.6f}"
        )

    both_within = both_hands["within_class"]["overall"]["mean"]
    both_between = both_hands["between_class"]["mean"]
    full_within = full_body["within_class"]["overall"]["mean"]
    full_between = full_body["between_class"]["mean"]
    if both_between > both_within:
        same_class_statement = (
            "For BOTH_HANDS, the aggregate mean between-class distance exceeds "
            "the aggregate mean within-class distance."
        )
    else:
        same_class_statement = (
            "For BOTH_HANDS, the aggregate mean between-class distance does not "
            "exceed the aggregate mean within-class distance."
        )
    if both_ratio is None or full_ratio is None or both_ratio == full_ratio:
        representation_statement = (
            "The measured between/within ratios do not distinguish one representation "
            "as having stronger separation."
        )
    elif both_ratio > full_ratio:
        representation_statement = (
            "BOTH_HANDS has the larger measured between/within ratio in this analysis."
        )
    else:
        representation_statement = (
            "POSE_PLUS_BOTH_HANDS has the larger measured between/within ratio "
            "in this analysis."
        )

    variability_sorted = sorted(
        both_hands["most_variable_classes"],
        key=lambda item: -item["within_class_mean_distance"],
    )
    closest_classes = both_hands[
        "class_relationships_sorted_by_closest_other_distance"
    ][:5]
    interpretation = {
        "same_vs_different_class_distances": same_class_statement,
        "highest_internal_variation_classes": [
            {
                "class": item["class"],
                "sentence": item["sentence"],
                "within_class_mean_distance": item["within_class_mean_distance"],
            }
            for item in variability_sorted[:5]
        ],
        "closest_class_relationships": [
            {
                "class": item["class"],
                "closest_other_class": item["closest_other_class"],
                "distance_to_closest_other_class": item[
                    "distance_to_closest_other_class"
                ],
            }
            for item in closest_classes
        ],
        "representation_comparison": representation_statement,
        "separation_strength_by_ratio": {
            "BOTH_HANDS": geometric_separation_label(both_ratio),
            "POSE_PLUS_BOTH_HANDS": geometric_separation_label(full_ratio),
            "heuristic_bands": {
                "weak": "ratio < 1.1",
                "moderate": "1.1 <= ratio < 1.5",
                "strong": "ratio >= 1.5",
            },
        },
        "caveat": (
            "Interpretation is based only on geometric DTW distance statistics "
            "from valid training samples; it does not select a neural architecture."
        ),
    }
    results["final_interpretation"] = interpretation

    print("\n" + "=" * 64)
    print("FINAL INTERPRETATION")
    print("=" * 64)
    print(same_class_statement)
    print("Highest internal variation in BOTH_HANDS (top 5):")
    for item in variability_sorted[:5]:
        print(
            f"  {item['class']} ({item['sentence']}): "
            f"within mean={item['within_class_mean_distance']:.6f}"
        )
    print("Closest class relationships in BOTH_HANDS (top 5):")
    for item in closest_classes:
        print(
            f"  {item['class']} -> {item['closest_other_class']}: "
            f"average distance={item['distance_to_closest_other_class']:.6f}"
        )
    print(representation_statement)
    print(
        "Geometric separation by between/within ratio: "
        f"BOTH_HANDS={geometric_separation_label(both_ratio)}, "
        f"POSE_PLUS_BOTH_HANDS={geometric_separation_label(full_ratio)}."
    )
    print(
        "Ratio bands used for descriptive labels: weak < 1.1, moderate 1.1-<1.5, "
        "strong >= 1.5."
    )
    print("These are distance statistics, not classification accuracy.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nJSON report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("\nRun command:")
    print("python scripts/intraclass_consistency.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
