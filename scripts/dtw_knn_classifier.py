"""DTW nearest-neighbor diagnostics for ISL landmark trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "processed" / "landmarks_manifest.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_DIMENSION = 225
LANDMARK_COUNT = 75
RESAMPLED_FRAMES = 32
DTW_WINDOW = 8
SEED = 42


def resolve_path(raw_path: str) -> Path:
    """Resolve a project-relative landmark path."""
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_frame(path: Path) -> pd.DataFrame:
    """Load a split and validate its required metadata."""
    frame = pd.read_csv(path, keep_default_na=False)
    missing = {"landmark_path", "label_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Split is empty: {path}")
    return frame


def load_sequence(raw_path: str, source: Path, row_index: int) -> np.ndarray:
    """Load one finite landmark sequence and reshape it to landmarks."""
    path = resolve_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Landmark file missing for {source} row {row_index}: {path}")
    try:
        values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not load {path}: {exc}") from exc
    if values.ndim != 2 or values.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"Expected (T, {FEATURE_DIMENSION}) in {path}, got {values.shape}")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError(f"Empty or non-finite landmarks in {path}")
    return values.reshape(values.shape[0], LANDMARK_COUNT, 3)


def resample(sequence: np.ndarray) -> np.ndarray:
    """Uniformly resample a sequence while preserving its complete time span."""
    source = np.linspace(0.0, 1.0, sequence.shape[0])
    target = np.linspace(0.0, 1.0, RESAMPLED_FRAMES)
    channels = [
        np.interp(target, source, sequence[:, landmark, coordinate])
        for landmark in range(sequence.shape[1])
        for coordinate in range(sequence.shape[2])
    ]
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


def load_mode_sequences(
    frame: pd.DataFrame, source: Path, start: int, stop: int
) -> list[np.ndarray]:
    """Load, resample, and select one landmark mode for all rows."""
    sequences: list[np.ndarray] = []
    for row_index, raw_path in enumerate(frame["landmark_path"]):
        sequence = resample(load_sequence(str(raw_path), source, row_index))
        sequences.append(sequence[:, start:stop].reshape(RESAMPLED_FRAMES, -1))
    return sequences


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return normalized Euclidean DTW distance within a temporal band."""
    rows, columns = first.shape[0], second.shape[0]
    costs = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0
    for row in range(1, rows + 1):
        start = max(1, row - DTW_WINDOW)
        end = min(columns, row + DTW_WINDOW)
        distances = np.linalg.norm(first[row - 1] - second[start - 1:end], axis=1)
        for column, distance in enumerate(distances, start=start):
            costs[row, column] = distance + min(
                costs[row - 1, column],
                costs[row, column - 1],
                costs[row - 1, column - 1],
            )
    if not np.isfinite(costs[rows, columns]):
        raise ValueError("DTW path is outside the configured window.")
    return float(costs[rows, columns] / (rows + columns))


def nearest_neighbors(
    query: np.ndarray, references: list[np.ndarray]
) -> np.ndarray:
    """Calculate distances from one query to every training sequence."""
    return np.asarray([dtw_distance(query, reference) for reference in references])


def vote_labels(
    distances: np.ndarray, labels: np.ndarray, k: int
) -> tuple[int, list[int]]:
    """Vote among the k closest samples and return ranked candidate classes."""
    order = np.argsort(distances, kind="stable")
    selected = order[:k]
    counts: dict[int, int] = {}
    distance_sums: dict[int, float] = {}
    for index in selected:
        label = int(labels[index])
        counts[label] = counts.get(label, 0) + 1
        distance_sums[label] = distance_sums.get(label, 0.0) + float(distances[index])
    ranked = sorted(
        counts,
        key=lambda label: (-counts[label], distance_sums[label], label),
    )
    return ranked[0], ranked


def top_k_labels(distances: np.ndarray, labels: np.ndarray, k: int) -> list[int]:
    """Return the k distinct classes ranked by nearest-neighbor distance."""
    order = np.argsort(distances, kind="stable")
    ranked: list[int] = []
    for index in order:
        label = int(labels[index])
        if label not in ranked:
            ranked.append(label)
        if len(ranked) == k:
            break
    return ranked


def evaluate(
    references: list[np.ndarray],
    reference_labels: np.ndarray,
    queries: list[np.ndarray],
    query_labels: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    """Evaluate 1/3/5-NN voting and nearest-neighbor ranking."""
    distances = np.vstack([nearest_neighbors(query, references) for query in queries])
    predictions: dict[int, np.ndarray] = {}
    for k in (1, 3, 5):
        predictions[k] = np.asarray(
            [vote_labels(row, reference_labels, k)[0] for row in distances]
        )
    ranking = [
        top_k_labels(row, reference_labels, min(5, len(class_labels)))
        for row in distances
    ]
    metrics: dict[str, Any] = {
        "1nn_accuracy": float(accuracy_score(query_labels, predictions[1])),
        "3nn_accuracy": float(accuracy_score(query_labels, predictions[3])),
        "5nn_accuracy": float(accuracy_score(query_labels, predictions[5])),
        "macro_f1": {
            f"{k}nn": float(
                f1_score(query_labels, predictions[k], average="macro", zero_division=0)
            )
            for k in (1, 3, 5)
        },
        "weighted_f1": {
            f"{k}nn": float(
                f1_score(query_labels, predictions[k], average="weighted", zero_division=0)
            )
            for k in (1, 3, 5)
        },
        "top_3_accuracy": float(
            np.mean([int(int(label) in candidates[:3]) for label, candidates in zip(query_labels, ranking)])
        ),
        "top_5_accuracy": float(
            np.mean([int(int(label) in candidates[:5]) for label, candidates in zip(query_labels, ranking)])
        ),
        "number_of_classes_correctly_recognized": int(
            np.sum(
                [
                    np.any(predictions[1][query_labels == class_index] == class_index)
                    for class_index in range(len(class_labels))
                    if np.any(query_labels == class_index)
                ]
            )
        ),
        "nearest_neighbor_distance": {
            "mean": float(np.min(distances, axis=1).mean()),
            "median": float(np.median(np.min(distances, axis=1))),
            "minimum": float(np.min(distances)),
            "maximum": float(np.max(np.min(distances, axis=1))),
        },
        "confusion_matrix": confusion_matrix(
            query_labels, predictions[1], labels=np.arange(len(class_labels))
        ).tolist(),
        "per_class_accuracy": {},
    }
    for class_index, class_label in enumerate(class_labels):
        mask = query_labels == class_index
        if np.any(mask):
            metrics["per_class_accuracy"][class_label] = float(
                accuracy_score(query_labels[mask], predictions[1][mask])
            )
    return metrics


def main() -> int:
    """Run the DTW nearest-neighbor diagnostic."""
    np.random.seed(SEED)
    train = load_frame(TRAIN_CSV)
    validation = load_frame(VAL_CSV)
    test = load_frame(TEST_CSV)
    manifest = load_frame(MANIFEST_CSV)
    manifest_paths = set(manifest["landmark_path"].astype(str))
    for frame, name in ((train, "train"), (validation, "validation"), (test, "test")):
        missing_paths = sorted(set(frame["landmark_path"].astype(str)) - manifest_paths)
        if missing_paths:
            raise ValueError(f"{name} references paths absent from the manifest: {missing_paths[:5]}")

    class_labels = sorted(train["label_id"].astype(str).unique())
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    all_frames = {"train": train, "validation": validation, "test": test}
    labels = {
        name: frame["label_id"].astype(str).map(class_to_index).to_numpy(np.int64)
        for name, frame in all_frames.items()
    }
    mode_ranges = {
        "BOTH_HANDS": (33, 75),
        "LEFT_HAND": (33, 54),
        "RIGHT_HAND": (54, 75),
    }
    results: dict[str, Any] = {
        "seed": SEED,
        "resampled_frames": RESAMPLED_FRAMES,
        "dtw_window": DTW_WINDOW,
        "samples": {name: len(frame) for name, frame in all_frames.items()},
        "num_classes": len(class_labels),
        "class_mapping": class_to_index,
        "results": {},
    }
    text: list[str] = [
        "DTW KNN CLASSIFIER DIAGNOSTIC",
        f"Seed: {SEED}",
        f"Resampled frames: {RESAMPLED_FRAMES}",
        f"DTW window: {DTW_WINDOW}",
        f"Samples: {results['samples']}",
        f"Classes: {len(class_labels)}",
        "",
    ]

    for mode, (start, stop) in mode_ranges.items():
        train_mode = load_mode_sequences(train, TRAIN_CSV, start, stop)
        validation_mode = load_mode_sequences(validation, VAL_CSV, start, stop)
        test_mode = load_mode_sequences(test, TEST_CSV, start, stop)
        scaler = StandardScaler().fit(np.vstack(train_mode))
        normalized_train = [scaler.transform(sequence).astype(np.float32) for sequence in train_mode]
        normalized_validation = [scaler.transform(sequence).astype(np.float32) for sequence in validation_mode]
        normalized_test = [scaler.transform(sequence).astype(np.float32) for sequence in test_mode]
        results["results"][mode] = {}
        for split, sequences, split_labels in (
            ("validation", normalized_validation, labels["validation"]),
            ("test", normalized_test, labels["test"]),
        ):
            results["results"][mode][split] = evaluate(
                normalized_train,
                labels["train"],
                sequences,
                split_labels,
                class_labels,
            )
            metrics = results["results"][mode][split]
            text.extend(
                [
                    mode,
                    split,
                    f"  1-NN accuracy: {metrics['1nn_accuracy']:.6f}",
                    f"  3-NN accuracy: {metrics['3nn_accuracy']:.6f}",
                    f"  5-NN accuracy: {metrics['5nn_accuracy']:.6f}",
                    f"  Macro F1: {json.dumps(metrics['macro_f1'], sort_keys=True)}",
                    f"  Weighted F1: {json.dumps(metrics['weighted_f1'], sort_keys=True)}",
                    f"  Top-3 accuracy: {metrics['top_3_accuracy']:.6f}",
                    f"  Top-5 accuracy: {metrics['top_5_accuracy']:.6f}",
                    f"  Classes correctly recognized: {metrics['number_of_classes_correctly_recognized']}",
                    f"  Nearest-neighbor distances: {json.dumps(metrics['nearest_neighbor_distance'], sort_keys=True)}",
                    "",
                ]
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dtw_knn_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "dtw_knn_results.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
