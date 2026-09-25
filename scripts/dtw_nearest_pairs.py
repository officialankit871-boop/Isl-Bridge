"""Inspect nearest DTW train pairs for validation and test split diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    """Load split or manifest metadata."""
    frame = pd.read_csv(path, keep_default_na=False)
    missing = {"landmark_path", "label_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Metadata file is empty: {path}")
    return frame


def load_sequence(raw_path: str, source: Path, row_index: int) -> np.ndarray:
    """Load one finite `(T, 225)` sequence and reshape it to `(T, 75, 3)`."""
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
    """Match dtw_knn_classifier.py's uniform 32-frame resampling."""
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
    """Load, resample, and select one hand mode."""
    sequences = []
    for row_index, raw_path in enumerate(frame["landmark_path"]):
        sequence = resample(load_sequence(str(raw_path), source, row_index))
        sequences.append(sequence[:, start:stop].reshape(RESAMPLED_FRAMES, -1))
    return sequences


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Match dtw_knn_classifier.py's bounded normalized Euclidean DTW."""
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


def nearest_distances(
    query: np.ndarray, references: list[np.ndarray]
) -> np.ndarray:
    """Calculate distances from one query to every training sequence."""
    return np.asarray([dtw_distance(query, reference) for reference in references])


def nearest_rows(
    references: list[np.ndarray],
    reference_frame: pd.DataFrame,
    queries: list[np.ndarray],
    query_frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Find each query's stable nearest training row and retain all distances."""
    rows: list[dict[str, Any]] = []
    all_distances: list[np.ndarray] = []
    for query_index, query in enumerate(queries):
        distances = nearest_distances(query, references)
        all_distances.append(distances)
        nearest_index = int(np.argmin(distances))
        query_row = query_frame.iloc[query_index]
        train_row = reference_frame.iloc[nearest_index]
        query_label = str(query_row["label_id"])
        train_label = str(train_row["label_id"])
        rows.append(
            {
                "query_index": query_index,
                "test_video_path": str(query_row.get("video_path", "")),
                "test_landmark_path": str(query_row["landmark_path"]),
                "test_class_name": str(query_row.get("class_name", query_label)),
                "test_label_id": query_label,
                "nearest_train_video_path": str(train_row.get("video_path", "")),
                "nearest_train_landmark_path": str(train_row["landmark_path"]),
                "nearest_train_class_name": str(train_row.get("class_name", train_label)),
                "nearest_train_label_id": train_label,
                "dtw_distance": float(distances[nearest_index]),
                "predicted_class_matches": bool(train_label == query_label),
            }
        )
    return rows, np.vstack(all_distances)


def distance_summary(
    rows: list[dict[str, Any]], distances: np.ndarray
) -> dict[str, Any]:
    """Summarize nearest pairs and exact/near match class agreement."""
    nearest = np.asarray([float(row["dtw_distance"]) for row in rows])
    same = np.asarray([bool(row["predicted_class_matches"]) for row in rows])
    thresholds: dict[str, Any] = {}
    for name, threshold, inclusive in (
        ("distance_eq_0", 0.0, True),
        ("distance_lt_0_01", 0.01, False),
        ("distance_lt_0_05", 0.05, False),
    ):
        mask = nearest == threshold if inclusive else nearest < threshold
        thresholds[name] = {
            "count": int(mask.sum()),
            "same_class_count": int(np.sum(mask & same)),
            "different_class_count": int(np.sum(mask & ~same)),
        }
    same_distances = nearest[same]
    different_distances = nearest[~same]
    return {
        "sample_count": len(rows),
        "nearest_neighbor_distance": {
            "mean": float(nearest.mean()),
            "median": float(np.median(nearest)),
            "minimum": float(nearest.min()),
            "maximum": float(nearest.max()),
        },
        "same_class_nearest_neighbor_count": int(same.sum()),
        "different_class_nearest_neighbor_count": int((~same).sum()),
        "same_class_nearest_neighbor_mean_distance": (
            float(same_distances.mean()) if same_distances.size else None
        ),
        "different_class_nearest_neighbor_mean_distance": (
            float(different_distances.mean()) if different_distances.size else None
        ),
        "nearest_neighbor_same_class_accuracy": float(same.mean()),
        "exact_and_near_matches": thresholds,
        "all_pairwise_distance_minimum": float(distances.min()),
        "all_pairwise_distance_maximum": float(distances.max()),
    }


def format_pair_lines(
    mode: str, split: str, rows: list[dict[str, Any]]
) -> list[str]:
    """Format every exact/near nearest pair for the text report."""
    lines = [f"{mode} {split} exact/near nearest pairs:"]
    for threshold_name, threshold, inclusive in (
        ("distance == 0", 0.0, True),
        ("distance < 0.01", 0.01, False),
    ):
        if inclusive:
            selected = [row for row in rows if float(row["dtw_distance"]) == threshold]
        else:
            selected = [row for row in rows if float(row["dtw_distance"]) < threshold]
        lines.append(f"  {threshold_name}: {len(selected)}")
        for row in selected:
            lines.append(
                f"    {row['test_landmark_path']} -> "
                f"{row['nearest_train_landmark_path']} | "
                f"distance={row['dtw_distance']:.9f} | "
                f"same_class={row['predicted_class_matches']}"
            )
    return lines


def main() -> int:
    """Run nearest-pair analysis for test and validation splits."""
    np.random.seed(SEED)
    train = load_frame(TRAIN_CSV)
    validation = load_frame(VAL_CSV)
    test = load_frame(TEST_CSV)
    manifest = load_frame(MANIFEST_CSV)
    manifest_paths = set(manifest["landmark_path"].astype(str))
    for frame, name in (
        (train, "train"),
        (validation, "validation"),
        (test, "test"),
    ):
        missing = sorted(set(frame["landmark_path"].astype(str)) - manifest_paths)
        if missing:
            raise ValueError(f"{name} references paths absent from manifest: {missing[:5]}")

    mode_ranges = {
        "BOTH_HANDS": (33, 75),
        "LEFT_HAND": (33, 54),
        "RIGHT_HAND": (54, 75),
    }
    results: dict[str, Any] = {
        "seed": SEED,
        "resampled_frames": RESAMPLED_FRAMES,
        "dtw_window": DTW_WINDOW,
        "samples": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "modes": {},
    }
    csv_rows_by_index: dict[int, dict[str, Any]] = {}
    text_lines = [
        "DTW NEAREST PAIRS DIAGNOSTIC",
        f"Seed: {SEED}",
        f"Preprocessing: uniform resampling to {RESAMPLED_FRAMES} frames; "
        f"training-only StandardScaler; DTW window {DTW_WINDOW}.",
        f"Samples: train={len(train)}, validation={len(validation)}, test={len(test)}",
        "",
    ]
    for mode, (start, stop) in mode_ranges.items():
        train_sequences = load_mode_sequences(train, TRAIN_CSV, start, stop)
        validation_sequences = load_mode_sequences(validation, VAL_CSV, start, stop)
        test_sequences = load_mode_sequences(test, TEST_CSV, start, stop)
        scaler = StandardScaler().fit(np.vstack(train_sequences))
        train_sequences = [scaler.transform(item).astype(np.float32) for item in train_sequences]
        validation_sequences = [
            scaler.transform(item).astype(np.float32) for item in validation_sequences
        ]
        test_sequences = [scaler.transform(item).astype(np.float32) for item in test_sequences]
        mode_results: dict[str, Any] = {}
        split_data = (
            ("validation", validation_sequences, validation),
            ("test", test_sequences, test),
        )
        for split, queries, query_frame in split_data:
            pair_rows, distances = nearest_rows(train_sequences, train, queries, query_frame)
            mode_results[split] = distance_summary(pair_rows, distances)
            mode_results[split]["pairs_sorted_by_distance"] = sorted(
                pair_rows, key=lambda row: (float(row["dtw_distance"]), row["query_index"])
            )
            text_lines.extend(format_pair_lines(mode, split, pair_rows))
            text_lines.append("")
            for row in pair_rows:
                if split == "test":
                    target = csv_rows_by_index.setdefault(
                        int(row["query_index"]),
                        {
                            "test_video_path": row["test_video_path"],
                            "test_landmark_path": row["test_landmark_path"],
                            "test_class_name": row["test_class_name"],
                            "test_label_id": row["test_label_id"],
                        },
                    )
                    prefix = mode.lower()
                    for key, value in row.items():
                        if key in {
                            "query_index",
                            "test_video_path",
                            "test_landmark_path",
                            "test_class_name",
                            "test_label_id",
                        }:
                            continue
                        target[f"{prefix}_{key}"] = value
        results["modes"][mode] = mode_results

    # Append the 20 closest test-to-train pairs per mode to the terminal report.
    for mode in mode_ranges:
        sorted_pairs = results["modes"][mode]["test"]["pairs_sorted_by_distance"][:20]
        text_lines.append(f"{mode} 20 closest test-to-train pairs:")
        for row in sorted_pairs:
            text_lines.append(
                f"  {row['test_landmark_path']} -> {row['nearest_train_landmark_path']} | "
                f"distance={row['dtw_distance']:.9f} | "
                f"test={row['test_class_name']} | train={row['nearest_train_class_name']} | "
                f"same_class={row['predicted_class_matches']}"
            )
        text_lines.append("")

    # The pair list is useful in JSON but does not need to be duplicated in the CSV.
    for mode_result in results["modes"].values():
        for split_result in mode_result.values():
            split_result.pop("pairs_sorted_by_distance", None)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [csv_rows_by_index[index] for index in sorted(csv_rows_by_index)]
    ).to_csv(REPORTS_DIR / "dtw_nearest_pairs.csv", index=False)
    (REPORTS_DIR / "dtw_nearest_pairs_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "dtw_nearest_pairs.txt").write_text(
        "\n".join(text_lines), encoding="utf-8"
    )
    print("\n".join(text_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
