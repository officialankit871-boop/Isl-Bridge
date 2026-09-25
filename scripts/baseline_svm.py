"""Train and evaluate a deterministic classical SVM baseline for ISL."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
INPUT_DIMENSION = 225
STATS_PER_FEATURE = 5


def resolve_landmark_path(raw_path: str) -> Path:
    """Resolve a split's project-relative landmark path."""
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_split(csv_path: Path) -> pd.DataFrame:
    """Read and validate split metadata required by this baseline."""
    frame = pd.read_csv(csv_path, keep_default_na=False)
    required = {"landmark_path", "label_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    if "feature_dim" in frame.columns:
        feature_dims = pd.to_numeric(frame["feature_dim"], errors="raise")
        if not (feature_dims == INPUT_DIMENSION).all():
            raise ValueError(
                f"Every feature_dim in {csv_path} must equal {INPUT_DIMENSION}."
            )
    if frame.empty:
        raise ValueError(f"Split {csv_path} is empty.")
    return frame


def extract_features(raw_path: str, split_path: Path, row_index: int) -> np.ndarray:
    """Extract five temporal statistics per landmark feature."""
    path = resolve_landmark_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Landmark file does not exist for {split_path} row {row_index}: {path}"
        )
    try:
        sequence = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not load landmark file {path}: {exc}") from exc

    if sequence.ndim != 2 or sequence.shape[1] != INPUT_DIMENSION:
        raise ValueError(
            f"Expected landmark shape (T, {INPUT_DIMENSION}) in {path}, "
            f"got {sequence.shape}"
        )
    if sequence.shape[0] == 0:
        raise ValueError(f"Landmark sequence is empty in {path}.")
    if not np.isfinite(sequence).all():
        raise ValueError(f"Non-finite landmark values found in {path}.")

    with np.errstate(over="raise", invalid="raise"):
        try:
            features = np.concatenate(
                (
                    sequence.mean(axis=0),
                    sequence.std(axis=0),
                    sequence.min(axis=0),
                    sequence.max(axis=0),
                    sequence[-1] - sequence[0],
                )
            ).astype(np.float32, copy=False)
        except FloatingPointError as exc:
            raise ValueError(f"Non-finite temporal statistics produced for {path}.") from exc
    if features.shape != (INPUT_DIMENSION * STATS_PER_FEATURE,):
        raise ValueError(f"Unexpected feature shape for {path}: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite temporal statistics produced for {path}.")
    return features


def build_features(frame: pd.DataFrame, split_path: Path) -> np.ndarray:
    """Extract fixed-length features for every row in a split."""
    return np.vstack(
        [
            extract_features(str(raw_path), split_path, row_index)
            for row_index, raw_path in enumerate(frame["landmark_path"])
        ]
    )


def top_k_accuracy(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Calculate top-k accuracy from SVC decision scores."""
    top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.any(top_indices == targets[:, None], axis=1).mean())


def evaluate(
    model: SVC,
    features: np.ndarray,
    targets: np.ndarray,
    class_labels: list[str],
) -> tuple[dict[str, float], str, np.ndarray]:
    """Evaluate a fitted model and return metrics, report text, and confusion matrix."""
    scores = np.asarray(model.decision_function(features))
    predictions = model.classes_[np.argmax(scores, axis=1)]
    label_indices = np.arange(len(class_labels))
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "top_3_accuracy": top_k_accuracy(scores, targets, min(3, len(class_labels))),
        "top_5_accuracy": top_k_accuracy(scores, targets, min(5, len(class_labels))),
    }
    report = classification_report(
        targets,
        predictions,
        labels=label_indices,
        target_names=class_labels,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=label_indices)
    return metrics, report, matrix


def save_reports(
    metrics: dict[str, object],
    reports: dict[str, str],
    matrices: dict[str, np.ndarray],
    class_labels: list[str],
) -> None:
    """Persist JSON metrics, text reports, and both confusion matrices."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "svm_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report_text = "\n".join(
        f"{split.upper()} CLASSIFICATION REPORT\n{'=' * 30}\n{report}"
        for split, report in reports.items()
    )
    (REPORTS_DIR / "svm_classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    for split, matrix in matrices.items():
        for true_index, row in enumerate(matrix):
            for predicted_index, count in enumerate(row):
                rows.append(
                    {
                        "split": split,
                        "true_label_id": class_labels[true_index],
                        "predicted_label_id": class_labels[predicted_index],
                        "count": int(count),
                    }
                )
    pd.DataFrame(rows).to_csv(
        REPORTS_DIR / "svm_confusion_matrix.csv", index=False
    )


def main() -> int:
    """Fit the SVM on training data and evaluate validation and test splits."""
    train_frame = load_split(TRAIN_CSV)
    val_frame = load_split(VAL_CSV)
    test_frame = load_split(TEST_CSV)
    class_labels = sorted(train_frame["label_id"].astype(str).unique())
    class_to_index = {label: index for index, label in enumerate(class_labels)}

    frames = {"train": train_frame, "validation": val_frame, "test": test_frame}
    split_paths = {"train": TRAIN_CSV, "validation": VAL_CSV, "test": TEST_CSV}
    for split, frame in frames.items():
        unknown = sorted(set(frame["label_id"].astype(str)) - set(class_to_index))
        if unknown:
            raise ValueError(
                f"{split} split contains label_id values absent from training: {unknown}"
            )

    extracted = {
        split: build_features(frame, split_paths[split])
        for split, frame in frames.items()
    }
    targets = {
        split: frame["label_id"].astype(str).map(class_to_index).to_numpy(dtype=np.int64)
        for split, frame in frames.items()
    }

    scaler = StandardScaler()
    model = SVC(
        kernel="rbf",
        probability=False,
        class_weight="balanced",
        random_state=42,
    )
    start_time = time.perf_counter()
    train_features = scaler.fit_transform(extracted["train"])
    model.fit(train_features, targets["train"])
    training_time = time.perf_counter() - start_time

    scaled = {
        split: scaler.transform(features)
        for split, features in extracted.items()
    }
    metrics: dict[str, object] = {
        "num_training_samples": len(train_frame),
        "num_validation_samples": len(val_frame),
        "num_test_samples": len(test_frame),
        "num_classes": len(class_labels),
        "feature_dimension": INPUT_DIMENSION * STATS_PER_FEATURE,
        "training_time_seconds": training_time,
        "class_mapping": class_to_index,
    }
    reports: dict[str, str] = {}
    matrices: dict[str, np.ndarray] = {}
    for split in ("validation", "test"):
        split_metrics, report, matrix = evaluate(
            model, scaled[split], targets[split], class_labels
        )
        metrics[split] = split_metrics
        reports[split] = report
        matrices[split] = matrix

    save_reports(metrics, reports, matrices, class_labels)
    print(f"Training samples: {len(train_frame)}")
    print(f"Validation samples: {len(val_frame)}")
    print(f"Test samples: {len(test_frame)}")
    print(f"Classes: {len(class_labels)}")
    print(f"Feature dimension: {INPUT_DIMENSION * STATS_PER_FEATURE}")
    print(f"Training time: {training_time:.3f} seconds")
    for split in ("validation", "test"):
        print(f"{split.capitalize()} metrics:")
        print(json.dumps(metrics[split], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
