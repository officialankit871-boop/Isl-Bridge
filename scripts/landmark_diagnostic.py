"""Diagnose ISL landmark separability with static/temporal statistics and KNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SPLIT_PATHS = {
    "train": PROJECT_ROOT / "data" / "splits" / "train.csv",
    "validation": PROJECT_ROOT / "data" / "splits" / "val.csv",
    "test": PROJECT_ROOT / "data" / "splits" / "test.csv",
}
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_manifest.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_DIMENSION = 225
SEED = 42


def resolve_path(raw_path: str) -> Path:
    """Resolve a project-relative path without modifying its source."""
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_metadata(path: Path) -> pd.DataFrame:
    """Load and validate split or manifest metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"landmark_path", "label_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Metadata file is empty: {path}")
    return frame


def load_sequence(raw_path: str, source: Path, row_index: int) -> np.ndarray:
    """Load one complete landmark sequence and enforce its contract."""
    path = resolve_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Landmark file does not exist for {source} row {row_index}: {path}"
        )
    try:
        sequence = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not load landmark file {path}: {exc}") from exc
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected shape (T, {FEATURE_DIMENSION}) in {path}, got {sequence.shape}"
        )
    if sequence.shape[0] == 0 or not np.isfinite(sequence).all():
        raise ValueError(f"Empty or non-finite landmark sequence in {path}")
    return sequence


def static_stats(sequence: np.ndarray) -> np.ndarray:
    """Return mean, std, min, max, and last-first for each feature."""
    features = np.concatenate(
        (
            sequence.mean(axis=0),
            sequence.std(axis=0),
            sequence.min(axis=0),
            sequence.max(axis=0),
            sequence[-1] - sequence[0],
        )
    ).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite static statistics were produced.")
    return features


def temporal_stats(sequence: np.ndarray) -> np.ndarray:
    """Return statistics for landmarks and their first and second differences."""
    first_difference = np.diff(sequence, n=1, axis=0)
    second_difference = np.diff(sequence, n=2, axis=0)
    parts: list[np.ndarray] = []
    for values in (sequence, first_difference, second_difference):
        if values.shape[0] == 0:
            parts.extend(
                (np.zeros(FEATURE_DIMENSION), np.zeros(FEATURE_DIMENSION),
                 np.zeros(FEATURE_DIMENSION), np.zeros(FEATURE_DIMENSION))
            )
        else:
            parts.extend(
                (values.mean(axis=0), values.std(axis=0),
                 values.min(axis=0), values.max(axis=0))
            )
    # These are explicitly retained as temporal mean/std features for each landmark.
    parts.extend((sequence.mean(axis=0), sequence.std(axis=0)))
    features = np.concatenate(parts).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite temporal statistics were produced.")
    return features


def build_features(
    frame: pd.DataFrame, source: Path, manifest_paths: set[str]
) -> dict[str, np.ndarray]:
    """Load every complete sequence and build both representations."""
    static_rows: list[np.ndarray] = []
    temporal_rows: list[np.ndarray] = []
    for row_index, raw_path in enumerate(frame["landmark_path"]):
        path_value = str(raw_path)
        if path_value not in manifest_paths:
            raise ValueError(f"{source} references a path absent from the manifest: {path_value}")
        sequence = load_sequence(path_value, source, row_index)
        static_rows.append(static_stats(sequence))
        temporal_rows.append(temporal_stats(sequence))
    return {
        "static_stats": np.vstack(static_rows),
        "temporal_stats": np.vstack(temporal_rows),
    }


def top_k_accuracy(probabilities: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Calculate top-k accuracy from KNN class probabilities."""
    top_indices = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.any(top_indices == targets[:, None], axis=1).mean())


def nearest_distance_diagnostics(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    eval_features: np.ndarray,
    eval_targets: np.ndarray,
) -> dict[str, float]:
    """Compare each evaluation sample with its closest same/different class sample."""
    distances = np.linalg.norm(
        eval_features[:, None, :] - train_features[None, :, :], axis=2
    )
    same = train_targets[None, :] == eval_targets[:, None]
    different = ~same
    same_distances = np.where(same, distances, np.inf).min(axis=1)
    different_distances = np.where(different, distances, np.inf).min(axis=1)
    nearest_indices = distances.argmin(axis=1)
    return {
        "same_class_nearest_neighbor_distance": float(same_distances.mean()),
        "different_class_nearest_neighbor_distance": float(
            different_distances.mean()
        ),
        "nearest_neighbor_accuracy": float(
            accuracy_score(eval_targets, train_targets[nearest_indices])
        ),
    }


def evaluate_knn(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    eval_features: np.ndarray,
    eval_targets: np.ndarray,
    class_count: int,
    neighbors: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Fit one KNN model on training features and evaluate one split."""
    model = KNeighborsClassifier(n_neighbors=neighbors, weights="uniform", n_jobs=1)
    model.fit(train_features, train_targets)
    predictions = model.predict(eval_features)
    probabilities = model.predict_proba(eval_features)
    # Align probabilities with the complete training-derived class index space.
    aligned = np.zeros((len(eval_features), class_count), dtype=np.float64)
    aligned[:, model.classes_] = probabilities
    metrics = {
        "accuracy": float(accuracy_score(eval_targets, predictions)),
        "macro_f1": float(
            f1_score(eval_targets, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(eval_targets, predictions, average="weighted", zero_division=0)
        ),
        "top_3_accuracy": top_k_accuracy(
            aligned, eval_targets, min(3, class_count)
        ),
        "top_5_accuracy": top_k_accuracy(
            aligned, eval_targets, min(5, class_count)
        ),
    }
    return metrics, predictions


def confused_pairs(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_labels: list[str],
) -> list[dict[str, Any]]:
    """Return the most frequent off-diagonal confusion pairs."""
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(class_labels)))
    pairs: list[dict[str, Any]] = []
    for true_index, predicted_index in zip(*np.nonzero(matrix)):
        if true_index == predicted_index:
            continue
        pairs.append(
            {
                "true_label_id": class_labels[true_index],
                "predicted_label_id": class_labels[predicted_index],
                "count": int(matrix[true_index, predicted_index]),
            }
        )
    return sorted(
        pairs,
        key=lambda item: (-int(item["count"]), item["true_label_id"], item["predicted_label_id"]),
    )[:15]


def main() -> int:
    """Run the deterministic landmark diagnostic."""
    np.random.seed(SEED)
    manifest = load_metadata(MANIFEST_PATH)
    manifest_paths = set(manifest["landmark_path"].astype(str))
    frames = {split: load_metadata(path) for split, path in SPLIT_PATHS.items()}
    class_labels = sorted(frames["train"]["label_id"].astype(str).unique())
    class_to_index = {label: index for index, label in enumerate(class_labels)}

    for split, frame in frames.items():
        unknown = sorted(set(frame["label_id"].astype(str)) - set(class_to_index))
        if unknown:
            raise ValueError(f"{split} contains labels absent from training: {unknown}")

    features = {
        split: build_features(frame, SPLIT_PATHS[split], manifest_paths)
        for split, frame in frames.items()
    }
    targets = {
        split: frame["label_id"].astype(str).map(class_to_index).to_numpy(np.int64)
        for split, frame in frames.items()
    }

    results: dict[str, Any] = {
        "seed": SEED,
        "samples": {split: len(frame) for split, frame in frames.items()},
        "num_classes": len(class_labels),
        "class_mapping": class_to_index,
        "representation_dimensions": {
            name: int(values.shape[1]) for name, values in features["train"].items()
        },
        "results": {},
        "most_confused_class_pairs": {},
    }
    output_text: list[str] = []
    for representation in ("static_stats", "temporal_stats"):
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(features["train"][representation])
        scaled_splits = {
            split: scaler.transform(features[split][representation])
            for split in ("validation", "test")
        }
        results["results"][representation] = {}
        confusion_targets: list[np.ndarray] = []
        confusion_predictions: list[np.ndarray] = []
        output_text.append(
            f"{representation.upper()} (dimension={scaled_train.shape[1]})"
        )
        for split in ("validation", "test"):
            results["results"][representation][split] = {}
            for neighbors in (1, 3, 5):
                metrics, predictions = evaluate_knn(
                    scaled_train,
                    targets["train"],
                    scaled_splits[split],
                    targets[split],
                    len(class_labels),
                    neighbors,
                )
                results["results"][representation][split][f"{neighbors}nn"] = metrics
                if neighbors == 1:
                    confusion_targets.append(targets[split])
                    confusion_predictions.append(predictions)
                output_text.append(
                    f"{split} {neighbors}-NN: {json.dumps(metrics, sort_keys=True)}"
                )
            results["results"][representation][split]["nearest_neighbor"] = (
                nearest_distance_diagnostics(
                    scaled_train,
                    targets["train"],
                    scaled_splits[split],
                    targets[split],
                )
            )
            output_text.append(
                f"{split} nearest-neighbor diagnostics: "
                f"{json.dumps(results['results'][representation][split]['nearest_neighbor'], sort_keys=True)}"
            )
        combined_targets = np.concatenate(confusion_targets)
        combined_predictions = np.concatenate(confusion_predictions)
        results["most_confused_class_pairs"][representation] = confused_pairs(
            combined_targets, combined_predictions, class_labels
        )
        output_text.append("15 most confused class pairs (1-NN, validation + test):")
        for pair in results["most_confused_class_pairs"][representation]:
            output_text.append(
                f"  {pair['true_label_id']} -> {pair['predicted_label_id']}: {pair['count']}"
            )
        output_text.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "landmark_diagnostic.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "landmark_diagnostic.txt").write_text(
        "\n".join(
            [
                "LANDMARK DIAGNOSTIC",
                f"Samples: {results['samples']}",
                f"Classes: {results['num_classes']}",
                f"Representation dimensions: {results['representation_dimensions']}",
                "",
                *output_text,
            ]
        ),
        encoding="utf-8",
    )
    print(f"Training samples: {len(frames['train'])}")
    print(f"Validation samples: {len(frames['validation'])}")
    print(f"Test samples: {len(frames['test'])}")
    print(f"Classes: {len(class_labels)}")
    print("\n".join(output_text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
