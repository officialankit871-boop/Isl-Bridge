"""Compare landmark component separability with Euclidean and DTW distances."""

from __future__ import annotations

import json
from collections import Counter
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
REPORT_PATH = PROJECT_ROOT / "reports" / "component_separability_report.json"

FEATURE_DIMENSION = 225
RESAMPLED_FRAMES = 32
DTW_WINDOW = 8
NEIGHBOR_COUNTS = (1, 3, 5)
TOP_K_VALUES = (3, 5)
DISPLAY_PAIR_COUNT = 10
SEED = 42

REPRESENTATIONS: dict[str, slice | tuple[slice, slice]] = {
    "POSE": slice(0, 99),
    "LEFT_HAND": slice(99, 162),
    "RIGHT_HAND": slice(162, 225),
    "BOTH_HANDS": slice(99, 225),
    "POSE_PLUS_BOTH_HANDS": (slice(0, 99), slice(99, 225)),
}


def resolve_path(value: str) -> Path:
    """Resolve relative data paths against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_manifest_sentences(path: Path) -> dict[str, str]:
    """Load a consistent label-to-sentence mapping from the manifest."""
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
    if not sentences:
        raise ValueError(f"Manifest contains no class mappings: {path}")
    return sentences


def resample_sequence(sequence: np.ndarray, frame_count: int = RESAMPLED_FRAMES) -> np.ndarray:
    """Linearly resample a sequence at evenly spaced normalized time positions."""
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected landmark shape (T, {FEATURE_DIMENSION}), got {sequence.shape}."
        )
    if sequence.shape[0] == 0:
        raise ValueError("Landmark sequence contains no frames.")

    positions = np.linspace(0, sequence.shape[0] - 1, frame_count, dtype=np.float64)
    left = np.floor(positions).astype(np.intp)
    right = np.minimum(left + 1, sequence.shape[0] - 1)
    alpha = (positions - left).astype(np.float32)[:, None]
    return (
        sequence[left] * (1.0 - alpha) + sequence[right] * alpha
    ).astype(np.float32, copy=False)


def load_split(
    split_name: str,
    path: Path,
) -> tuple[list[np.ndarray], list[str], dict[str, int], list[dict[str, str]]]:
    """Read split sequences, removing invalid frames and recording bad rows."""
    if not path.is_file():
        raise FileNotFoundError(f"{split_name} split not found: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"landmark_path", "label_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    sequences: list[np.ndarray] = []
    labels: list[str] = []
    issues: list[dict[str, str]] = []
    invalid_frame_count = 0

    for row_index, row in frame.iterrows():
        label = str(row["label_id"]).strip()
        raw_path = str(row["landmark_path"]).strip()
        landmark_path = resolve_path(raw_path)
        try:
            if not label:
                raise ValueError("Empty label_id.")
            if not raw_path:
                raise ValueError("Empty landmark_path.")
            values = np.load(landmark_path, allow_pickle=False)
            if values.ndim != 2 or values.shape[1] != FEATURE_DIMENSION:
                raise ValueError(
                    f"Expected shape (T, {FEATURE_DIMENSION}), got {values.shape}."
                )
            if values.shape[0] == 0:
                raise ValueError("Sequence has zero frames.")
            try:
                values = np.asarray(values, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Landmark data is not numeric: {exc}") from exc

            valid_frames = np.isfinite(values).all(axis=1)
            invalid_count = int((~valid_frames).sum())
            if invalid_count:
                invalid_frame_count += invalid_count
                print(
                    f"Warning: {split_name} row {row_index + 2}: removing "
                    f"{invalid_count} invalid frame(s) from {landmark_path}."
                )
                values = values[valid_frames]
            if values.shape[0] == 0:
                raise ValueError("No finite frames remain after invalid frames were removed.")

            sequences.append(resample_sequence(values))
            labels.append(label)
        except (OSError, ValueError, TypeError, EOFError) as exc:
            issue = {
                "split": split_name,
                "row_number": str(row_index + 2),
                "label_id": label,
                "landmark_path": str(landmark_path),
                "error": str(exc),
            }
            issues.append(issue)
            print(f"Warning: skipping {split_name} row {row_index + 2}: {exc}")

    if not sequences:
        raise ValueError(f"No valid samples loaded from {split_name} split {path}.")
    summary = {
        "rows_in_split": int(len(frame)),
        "samples_used": len(sequences),
        "classes_used": len(set(labels)),
        "invalid_frames_removed": invalid_frame_count,
        "samples_skipped": len(issues),
    }
    return sequences, labels, summary, issues


def select_components(
    sequences: list[np.ndarray],
    component: slice | tuple[slice, slice],
) -> np.ndarray:
    """Select one representation and return samples x frames x features."""
    if isinstance(component, tuple):
        selected = np.stack(
            [
                np.concatenate([sequence[:, part] for part in component], axis=1)
                for sequence in sequences
            ]
        )
    else:
        selected = np.stack([sequence[:, component] for sequence in sequences])
    if selected.ndim != 3 or not np.isfinite(selected).all():
        raise ValueError("Selected component data must be finite with shape (N, 32, D).")
    return selected.astype(np.float64, copy=False)


def scale_flattened(
    train_sequences: np.ndarray,
    validation_sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a flattened-sequence scaler on training samples only."""
    train_flat = train_sequences.reshape(len(train_sequences), -1)
    validation_flat = validation_sequences.reshape(len(validation_sequences), -1)
    scaler = StandardScaler()
    return scaler.fit_transform(train_flat), scaler.transform(validation_flat)


def scale_frames(
    train_sequences: np.ndarray,
    validation_sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a per-feature scaler on training frames only for DTW inputs."""
    feature_dim = train_sequences.shape[2]
    scaler = StandardScaler()
    scaler.fit(train_sequences.reshape(-1, feature_dim))
    train_scaled = scaler.transform(
        train_sequences.reshape(-1, feature_dim)
    ).reshape(train_sequences.shape)
    validation_scaled = scaler.transform(
        validation_sequences.reshape(-1, feature_dim)
    ).reshape(validation_sequences.shape)
    return train_scaled, validation_scaled


def classify_from_neighbors(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    train_labels: list[str],
    validation_labels: list[str],
) -> dict[str, Any]:
    """Calculate kNN majority accuracy, top-k neighbor accuracy, and 1-NN F1."""
    neighbor_labels = np.asarray(train_labels, dtype=object)[neighbor_indices]
    expected = np.asarray(validation_labels, dtype=object)
    result: dict[str, Any] = {}
    for neighbor_count in NEIGHBOR_COUNTS:
        predictions: list[str] = []
        for row_labels, row_distances in zip(
            neighbor_labels[:, :neighbor_count],
            neighbor_distances[:, :neighbor_count],
        ):
            counts = Counter(row_labels.tolist())
            first_distance = {
                label: float(row_distances[np.flatnonzero(row_labels == label)[0]])
                for label in counts
            }
            prediction = min(
                counts,
                key=lambda label: (-counts[label], first_distance[label], label),
            )
            predictions.append(prediction)
        result[f"{neighbor_count}nn_accuracy"] = float(
            np.mean(expected == np.asarray(predictions, dtype=object))
        )
        if neighbor_count == 1:
            predicted = np.asarray(predictions, dtype=object)
            labels = sorted(set(train_labels) | set(validation_labels))
            class_f1 = []
            for label in labels:
                true_positive = int(np.sum((expected == label) & (predicted == label)))
                false_positive = int(np.sum((expected != label) & (predicted == label)))
                false_negative = int(np.sum((expected == label) & (predicted != label)))
                denominator = 2 * true_positive + false_positive + false_negative
                class_f1.append(
                    2 * true_positive / denominator if denominator else 0.0
                )
            result["1nn_macro_f1"] = float(np.mean(class_f1))

    for top_k in TOP_K_VALUES:
        top_k_correct = [
            actual in row_labels[:top_k]
            for actual, row_labels in zip(validation_labels, neighbor_labels)
        ]
        result[f"top{top_k}_accuracy"] = float(np.mean(top_k_correct))

    result["nearest_same_class_percentage"] = float(
        100.0 * np.mean(neighbor_labels[:, 0] == np.asarray(validation_labels))
    )
    result["mean_nearest_distance"] = float(np.mean(neighbor_distances[:, 0]))
    result["median_nearest_distance"] = float(np.median(neighbor_distances[:, 0]))
    return result


def nearest_neighbor_pairs(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    train_labels: list[str],
    validation_labels: list[str],
    sentence_by_label: dict[str, str],
) -> list[dict[str, Any]]:
    """Return the ten validation-to-nearest-training records ordered by distance."""
    pairs: list[dict[str, Any]] = []
    for validation_index, (train_index, distance) in enumerate(
        zip(neighbor_indices[:, 0], neighbor_distances[:, 0])
    ):
        validation_class = validation_labels[validation_index]
        training_class = train_labels[int(train_index)]
        pairs.append(
            {
                "validation_class": validation_class,
                "validation_sentence": sentence_by_label.get(validation_class),
                "training_class": training_class,
                "training_sentence": sentence_by_label.get(training_class),
                "distance": float(distance),
                "same_class": validation_class == training_class,
                "_validation_index": validation_index,
            }
        )
    pairs.sort(
        key=lambda pair: (
            pair["distance"],
            pair["validation_class"],
            pair["_validation_index"],
        )
    )
    for pair in pairs:
        pair.pop("_validation_index")
    return pairs[:DISPLAY_PAIR_COUNT]


def euclidean_neighbors(
    train_features: np.ndarray,
    validation_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable-sorted nearest training indices and distances."""
    distances = euclidean_distances(validation_features, train_features)
    indices = np.argsort(distances, axis=1, kind="stable")
    return indices, np.take_along_axis(distances, indices, axis=1)


def dtw_distances_to_training(
    query: np.ndarray,
    candidates: np.ndarray,
    window: int = DTW_WINDOW,
) -> np.ndarray:
    """Compute band-limited Euclidean-cost DTW to all candidates using NumPy."""
    if query.ndim != 2 or candidates.ndim != 3:
        raise ValueError("DTW expects query (T,D) and candidates (N,T,D).")
    if candidates.shape[1:] != query.shape:
        raise ValueError(
            f"DTW sequence shapes must match; got {query.shape} and {candidates.shape[1:]}."
        )

    frame_count = query.shape[0]
    candidate_count = candidates.shape[0]
    band = max(window, abs(frame_count - candidates.shape[1]))
    previous = np.full((frame_count + 1, candidate_count), np.inf, dtype=np.float64)
    previous[0] = 0.0

    for query_index in range(1, frame_count + 1):
        current = np.full((frame_count + 1, candidate_count), np.inf, dtype=np.float64)
        start = max(1, query_index - band)
        end = min(frame_count, query_index + band)
        for candidate_index in range(start, end + 1):
            differences = candidates[:, candidate_index - 1, :] - query[query_index - 1]
            frame_costs = np.sqrt(np.einsum("nd,nd->n", differences, differences))
            best_previous = np.minimum(
                previous[candidate_index],
                np.minimum(current[candidate_index - 1], previous[candidate_index - 1]),
            )
            current[candidate_index] = frame_costs + best_previous
        previous = current
    return previous[frame_count]


def dtw_neighbor_search(
    train_sequences: np.ndarray,
    validation_sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute all validation-to-training DTW distances, CPU-only."""
    all_distances = np.empty(
        (len(validation_sequences), len(train_sequences)),
        dtype=np.float64,
    )
    for validation_index, sequence in enumerate(validation_sequences):
        all_distances[validation_index] = dtw_distances_to_training(
            sequence,
            train_sequences,
        )
        if (validation_index + 1) % 10 == 0 or validation_index + 1 == len(validation_sequences):
            print(
                f"  DTW progress: {validation_index + 1}/"
                f"{len(validation_sequences)} validation samples"
            )
    neighbor_indices = np.argsort(all_distances, axis=1, kind="stable")
    neighbor_distances = np.take_along_axis(
        all_distances,
        neighbor_indices,
        axis=1,
    )
    return neighbor_indices, neighbor_distances


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print classification and nearest-distance metrics."""
    print(f"1-NN accuracy: {metrics['1nn_accuracy']:.4f}")
    print(f"3-NN accuracy: {metrics['3nn_accuracy']:.4f}")
    print(f"5-NN accuracy: {metrics['5nn_accuracy']:.4f}")
    print(f"Top-3 accuracy: {metrics['top3_accuracy']:.4f}")
    print(f"Top-5 accuracy: {metrics['top5_accuracy']:.4f}")
    if "1nn_macro_f1" in metrics:
        print(f"1-NN macro F1: {metrics['1nn_macro_f1']:.4f}")
    print(
        f"Nearest same-class percentage: "
        f"{metrics['nearest_same_class_percentage']:.2f}%"
    )
    print(f"Mean nearest distance: {metrics['mean_nearest_distance']:.6f}")
    print(f"Median nearest distance: {metrics['median_nearest_distance']:.6f}")


def print_pairs(pairs: list[dict[str, Any]]) -> None:
    """Print the ten closest validation-to-training sample pairs."""
    for rank, pair in enumerate(pairs, start=1):
        validation_sentence = pair["validation_sentence"] or "<unmapped>"
        training_sentence = pair["training_sentence"] or "<unmapped>"
        print(
            f"{rank:2d}. validation={pair['validation_class']} "
            f'("{validation_sentence}") -> training={pair["training_class"]} '
            f'("{training_sentence}") | distance={pair["distance"]:.6f} | '
            f"same_class={str(pair['same_class']).lower()}"
        )


def analyze_representation(
    name: str,
    component: slice | tuple[slice, slice],
    train_sequences: list[np.ndarray],
    validation_sequences: list[np.ndarray],
    train_labels: list[str],
    validation_labels: list[str],
    sentence_by_label: dict[str, str],
    method: str,
) -> dict[str, Any]:
    """Run one representation through its train-scaled distance experiment."""
    if method not in {"euclidean_flattened", "dtw_temporal"}:
        raise ValueError(f"Unsupported analysis method: {method}")
    train_component = select_components(train_sequences, component)
    validation_component = select_components(validation_sequences, component)

    if method == "euclidean_flattened":
        train_scaled, validation_scaled = scale_flattened(
            train_component,
            validation_component,
        )
        neighbor_indices, sorted_distances = euclidean_neighbors(
            train_scaled,
            validation_scaled,
        )
    else:
        train_scaled, validation_scaled = scale_frames(
            train_component,
            validation_component,
        )
        neighbor_indices, sorted_distances = dtw_neighbor_search(
            train_scaled,
            validation_scaled,
        )

    metrics = classify_from_neighbors(
        neighbor_indices,
        sorted_distances,
        train_labels,
        validation_labels,
    )
    pairs = nearest_neighbor_pairs(
        neighbor_indices,
        sorted_distances,
        train_labels,
        validation_labels,
        sentence_by_label,
    )
    print(f"\nRepresentation: {name}")
    print(f"Feature dimension per frame: {train_component.shape[2]}")
    print_metrics(metrics)
    print("10 closest validation -> training pairs:")
    print_pairs(pairs)

    if method == "euclidean_flattened":
        return {
            "feature_dimension_per_frame": int(train_component.shape[2]),
            method: {
                "metrics": metrics,
                "closest_validation_pairs": pairs,
                "scaler_fit": "training samples only",
            },
        }
    return {
        method: {
            "window": DTW_WINDOW,
            "frame_cost": "Euclidean",
            "scaler_fit": "training frames only",
            "metrics": metrics,
            "closest_validation_pairs": pairs,
        }
    }


def main() -> int:
    """Run CPU-only component diagnostics and persist results as JSON."""
    np.random.seed(SEED)
    sentence_by_label = load_manifest_sentences(MANIFEST_PATH)
    train_sequences, train_labels, train_summary, train_issues = load_split(
        "TRAIN",
        TRAIN_PATH,
    )
    validation_sequences, validation_labels, validation_summary, validation_issues = load_split(
        "VALIDATION",
        VALIDATION_PATH,
    )

    print("============================================================")
    print("ISL BRIDGE COMPONENT SEPARABILITY DIAGNOSTIC")
    print("============================================================")
    print("\nDATASET")
    print(f"Seed: {SEED}")
    print(f"Device: CPU only")
    print(f"Manifest classes: {len(sentence_by_label)}")
    print(
        f"Train samples: {train_summary['samples_used']} used / "
        f"{train_summary['rows_in_split']} rows; classes={train_summary['classes_used']}"
    )
    print(
        f"Validation samples: {validation_summary['samples_used']} used / "
        f"{validation_summary['rows_in_split']} rows; "
        f"classes={validation_summary['classes_used']}"
    )
    print(
        f"Invalid frames removed: train={train_summary['invalid_frames_removed']}, "
        f"validation={validation_summary['invalid_frames_removed']}"
    )
    print(
        f"Missing/corrupt samples skipped: train={train_summary['samples_skipped']}, "
        f"validation={validation_summary['samples_skipped']}"
    )
    print("Only train.csv and val.csv are used; no test samples are loaded.")
    print("All representations use deterministic 32-frame linear resampling.")

    report: dict[str, Any] = {
        "dataset": {
            "seed": SEED,
            "manifest_class_count": len(sentence_by_label),
            "landmark_dimension": FEATURE_DIMENSION,
            "resampled_frames": RESAMPLED_FRAMES,
            "train": train_summary,
            "validation": validation_summary,
            "issues": train_issues + validation_issues,
        },
        "representations": {},
        "interpretation": (
            "Distance and nearest-neighbor diagnostics describe these representations "
            "on the selected train/validation data only; they do not select an "
            "architecture or establish universal superiority."
        ),
    }
    compact_rows: list[tuple[str, float, float, float, float]] = []
    print("\n============================================================")
    print("EUCLIDEAN / FLATTENED FEATURES")
    print("============================================================")
    for representation_name, component_slice in REPRESENTATIONS.items():
        result = analyze_representation(
            representation_name,
            component_slice,
            train_sequences,
            validation_sequences,
            train_labels,
            validation_labels,
            sentence_by_label,
            "euclidean_flattened",
        )
        report["representations"][representation_name] = result

    print("\n============================================================")
    print("DTW TEMPORAL FEATURES")
    print("============================================================")
    print(
        f"DTW window={DTW_WINDOW}; Euclidean frame cost; "
        "frame-wise StandardScaler fitted on train frames only."
    )
    for representation_name, result in report["representations"].items():
        component_slice = REPRESENTATIONS[representation_name]
        dtw_result = analyze_representation(
            representation_name,
            component_slice,
            train_sequences,
            validation_sequences,
            train_labels,
            validation_labels,
            sentence_by_label,
            "dtw_temporal",
        )
        result.update(dtw_result)
        dtw_metrics = dtw_result["dtw_temporal"]["metrics"]
        compact_rows.append(
            (
                representation_name,
                result["euclidean_flattened"]["metrics"]["1nn_accuracy"],
                result["euclidean_flattened"]["metrics"][
                    "nearest_same_class_percentage"
                ],
                dtw_metrics["1nn_accuracy"],
                dtw_metrics["nearest_same_class_percentage"],
            )
        )

    print("\n============================================================")
    print("FINAL COMPARISON")
    print("============================================================")
    print(
        "Representation | Euclidean 1NN | Euclidean same-class % | "
        "DTW 1NN | DTW same-class %"
    )
    print("-" * 91)
    for name, euclidean_accuracy, euclidean_same, dtw_accuracy, dtw_same in compact_rows:
        print(
            f"{name:<24} | {euclidean_accuracy:>13.4f} | "
            f"{euclidean_same:>21.2f}% | {dtw_accuracy:>7.4f} | {dtw_same:>16.2f}%"
        )
    print(
        "\nNeutral interpretation: compare the measured values across these "
        "representations and distance methods. No representation is universally "
        "better based on this diagnostic alone."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nJSON report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("\nRun command:")
    print("python scripts/component_separability.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
