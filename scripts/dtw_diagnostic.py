"""Diagnose temporal landmark similarity with bounded, deterministic DTW."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "processed" / "landmarks_manifest.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_DIMENSION = 225
LANDMARK_COUNT = 75
MAX_FRAMES = 96
DTW_WINDOW = 24
SEED = 42
CLASS_COUNT = 20
SAMPLES_PER_CLASS = 3


def resolve_path(raw_path: str) -> Path:
    """Resolve a project-relative landmark path."""
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_sequence(raw_path: str, row_index: int) -> np.ndarray:
    """Load and validate one full landmark sequence."""
    path = resolve_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Landmark file missing for row {row_index}: {path}")
    try:
        sequence = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not load {path}: {exc}") from exc
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected shape (T, {FEATURE_DIMENSION}) in {path}, got {sequence.shape}"
        )
    if sequence.shape[0] == 0 or not np.isfinite(sequence).all():
        raise ValueError(f"Empty or non-finite sequence in {path}")
    return sequence.reshape(sequence.shape[0], LANDMARK_COUNT, 3)


def resample(sequence: np.ndarray) -> np.ndarray:
    """Resample a sequence to a bounded length while preserving its time span."""
    source = np.linspace(0.0, 1.0, sequence.shape[0])
    target = np.linspace(0.0, 1.0, MAX_FRAMES)
    return np.stack(
        [
            np.interp(target, source, sequence[:, landmark, axis])
            for landmark in range(sequence.shape[1])
            for axis in range(sequence.shape[2])
        ],
        axis=1,
    ).reshape(MAX_FRAMES, sequence.shape[1], sequence.shape[2])


def hand_features(sequence: np.ndarray, start: int) -> np.ndarray:
    """Build compact hand coordinates, wrist trajectory, and wrist geometry."""
    hand = sequence[:, start : start + 21]
    wrist = hand[:, 0, :]
    distances = np.linalg.norm(hand - wrist[:, None, :], axis=2)
    return np.concatenate((hand.reshape(hand.shape[0], -1), wrist, distances), axis=1)


def pose_features(sequence: np.ndarray) -> np.ndarray:
    """Build pose coordinates plus shoulder-relative geometry."""
    pose = sequence[:, :33]
    center = (pose[:, 11] + pose[:, 12]) / 2.0
    distances = np.linalg.norm(pose - center[:, None, :], axis=2)
    return np.concatenate((pose.reshape(pose.shape[0], -1), center, distances), axis=1)


def representations(sequence: np.ndarray) -> dict[str, np.ndarray]:
    """Return the four requested per-frame representations."""
    sequence = resample(sequence)
    left = hand_features(sequence, 33)
    right = hand_features(sequence, 54)
    return {
        "both_hands": np.concatenate((left, right), axis=1),
        "left_hand": left,
        "right_hand": right,
        "pose": pose_features(sequence),
    }


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate normalized Euclidean DTW distance within a Sakoe-Chiba band."""
    rows, columns = first.shape[0], second.shape[0]
    infinity = np.inf
    costs = np.full((rows + 1, columns + 1), infinity, dtype=np.float64)
    costs[0, 0] = 0.0
    for row in range(1, rows + 1):
        start = max(1, row - DTW_WINDOW)
        end = min(columns, row + DTW_WINDOW)
        distances = np.linalg.norm(first[row - 1] - second[start - 1 : end], axis=1)
        for offset, distance in enumerate(distances, start=start):
            costs[row, offset] = distance + min(
                costs[row - 1, offset],
                costs[row, offset - 1],
                costs[row - 1, offset - 1],
            )
    if not np.isfinite(costs[rows, columns]):
        raise ValueError("DTW path was outside the configured window.")
    return float(costs[rows, columns] / (rows + columns))


def summarize(values: list[float]) -> dict[str, float | int]:
    """Summarize a collection of pairwise distances."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "pair_count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> int:
    """Run the sampled same-class and different-class DTW comparison."""
    train = pd.read_csv(TRAIN_CSV, keep_default_na=False)
    manifest = pd.read_csv(MANIFEST_CSV, keep_default_na=False)
    required = {"landmark_path", "label_id"}
    for frame, path in ((train, TRAIN_CSV), (manifest, MANIFEST_CSV)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    manifest_paths = set(manifest["landmark_path"].astype(str))
    train = train[train["landmark_path"].astype(str).isin(manifest_paths)].copy()
    counts = train["label_id"].astype(str).value_counts()
    eligible = sorted(counts[counts >= SAMPLES_PER_CLASS].index)
    if len(eligible) < CLASS_COUNT:
        raise ValueError(f"Only {len(eligible)} classes have at least five samples.")

    rng = np.random.default_rng(SEED)
    selected_classes = sorted(
        rng.choice(eligible, size=CLASS_COUNT, replace=False).tolist()
    )
    sequences: dict[str, dict[str, np.ndarray]] = {}
    selected_paths: dict[str, list[str]] = {}
    for label in selected_classes:
        class_rows = train[train["label_id"].astype(str) == label]
        chosen_indices = rng.choice(len(class_rows), size=SAMPLES_PER_CLASS, replace=False)
        selected_paths[label] = []
        for sample_number, position in enumerate(chosen_indices):
            row = class_rows.iloc[int(position)]
            path = str(row["landmark_path"])
            sequence = load_sequence(path, int(row.name))
            sequences[f"{label}:{sample_number}"] = representations(sequence)
            selected_paths[label].append(path)

    results: dict[str, Any] = {
        "seed": SEED,
        "selected_classes": selected_classes,
        "samples_per_class": SAMPLES_PER_CLASS,
        "selected_paths": selected_paths,
        "max_frames": MAX_FRAMES,
        "dtw_window": DTW_WINDOW,
        "representations": {},
    }
    text_lines = [
        "DTW LANDMARK DIAGNOSTIC",
        f"Seed: {SEED}",
        f"Selected classes: {len(selected_classes)}",
        f"Samples per class: {SAMPLES_PER_CLASS}",
        f"Feature sequence maximum length: {MAX_FRAMES}",
        f"DTW window: {DTW_WINDOW}",
        "",
    ]
    for representation in ("both_hands", "left_hand", "right_hand", "pose"):
        same_distances: list[float] = []
        different_distances: list[float] = []
        keys = list(sequences)
        key_labels = {key: key.split(":", 1)[0] for key in keys}
        for index, first_key in enumerate(keys):
            for second_key in keys[index + 1 :]:
                distance = dtw_distance(
                    sequences[first_key][representation],
                    sequences[second_key][representation],
                )
                if key_labels[first_key] == key_labels[second_key]:
                    same_distances.append(distance)
                else:
                    different_distances.append(distance)
        same_summary = summarize(same_distances)
        different_summary = summarize(different_distances)
        ratio = same_summary["mean"] / different_summary["mean"]
        results["representations"][representation] = {
            "same_class": same_summary,
            "different_class": different_summary,
            "mean_same_to_mean_different_ratio": float(ratio),
            "interpretation": (
                "same-class trajectories are closer"
                if ratio < 1.0
                else "same-class trajectories are farther apart"
            ),
        }
        text_lines.extend(
            [
                representation.upper(),
                f"  Same-class pairs: {same_summary['pair_count']}",
                f"  Different-class pairs: {different_summary['pair_count']}",
                f"  Same-class mean/median/min/max: "
                f"{same_summary['mean']:.6f} / {same_summary['median']:.6f} / "
                f"{same_summary['minimum']:.6f} / {same_summary['maximum']:.6f}",
                f"  Different-class mean/median/min/max: "
                f"{different_summary['mean']:.6f} / {different_summary['median']:.6f} / "
                f"{different_summary['minimum']:.6f} / {different_summary['maximum']:.6f}",
                f"  Mean same / mean different: {ratio:.6f}",
                f"  Interpretation: {results['representations'][representation]['interpretation']}",
                "",
            ]
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dtw_diagnostic.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "dtw_diagnostic.txt").write_text(
        "\n".join(text_lines), encoding="utf-8"
    )
    print("\n".join(text_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
