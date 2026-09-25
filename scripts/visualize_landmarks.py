"""Visualize representative ISL landmark sequences without training a model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "processed" / "landmarks_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "landmark_visualization"
SUMMARY_PATH = OUTPUT_DIR / "landmark_visualization_summary.txt"
FEATURE_DIMENSION = 225
LANDMARK_COUNT = 75
SEED = 42
SELECTED_CLASS_COUNT = 3
SAMPLES_PER_CLASS = 3


def resolve_path(raw_path: str) -> Path:
    """Resolve project-relative landmark paths."""
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_sequence(raw_path: str, row_index: int) -> np.ndarray:
    """Load and validate a complete landmark sequence."""
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


def summarize(sequence: np.ndarray) -> dict[str, float | int]:
    """Calculate presence and coordinate statistics for one sequence."""
    groups = {
        "pose": sequence[:, 0:33],
        "left_hand": sequence[:, 33:54],
        "right_hand": sequence[:, 54:75],
    }
    summary: dict[str, float | int] = {"sequence_length": int(sequence.shape[0])}
    for name, values in groups.items():
        summary[f"{name}_zero_ratio"] = float(np.mean(values == 0.0))
        summary[f"{name}_presence_ratio"] = float(np.mean(values != 0.0))
    coordinates = sequence.reshape(-1)
    summary.update(
        {
            "coordinate_min": float(coordinates.min()),
            "coordinate_max": float(coordinates.max()),
            "coordinate_mean": float(coordinates.mean()),
            "coordinate_std": float(coordinates.std()),
        }
    )
    return summary


def plot_sample(
    sequence: np.ndarray,
    class_label: str,
    sample_number: int,
    file_name: str,
) -> Path:
    """Save a four-panel 2D trajectory plot for one sample."""
    output_path = OUTPUT_DIR / f"{class_label}_sample_{sample_number}.png"
    panels = (
        ("Left wrist", sequence[:, 33, :2], "tab:blue"),
        ("Right wrist", sequence[:, 54, :2], "tab:orange"),
        ("Left hand landmarks", sequence[:, 33:54, :2], "tab:green"),
        ("Right hand landmarks", sequence[:, 54:75, :2], "tab:red"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, (title, values, color) in zip(axes.flat, panels):
        if values.ndim == 2:
            axis.plot(values[:, 0], values[:, 1], color=color, linewidth=1.5)
            axis.scatter(values[0, 0], values[0, 1], color="black", label="start", s=25)
            axis.scatter(values[-1, 0], values[-1, 1], color="gold", label="end", s=25)
        else:
            axis.scatter(
                values[:, :, 0].ravel(),
                values[:, :, 1].ravel(),
                c=np.tile(np.arange(values.shape[0]), values.shape[1]),
                cmap="viridis",
                s=4,
                alpha=0.55,
            )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.25)
        axis.invert_yaxis()
    figure.suptitle(f"{class_label} - sample {sample_number}\n{file_name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def compare_class_samples(samples: list[dict[str, float | int]]) -> str:
    """Describe whether the selected samples have similar lengths/statistics."""
    lengths = np.array([float(sample["sequence_length"]) for sample in samples])
    zero_ratios = np.array(
        [
            [
                float(sample["left_hand_zero_ratio"]),
                float(sample["right_hand_zero_ratio"]),
                float(sample["pose_zero_ratio"]),
            ]
            for sample in samples
        ]
    )
    coordinate_values = np.array(
        [
            [
                float(sample["coordinate_mean"]),
                float(sample["coordinate_std"]),
            ]
            for sample in samples
        ]
    )
    length_similar = (lengths.max() - lengths.min()) <= 0.25 * lengths.mean()
    zero_similar = np.all(zero_ratios.max(axis=0) - zero_ratios.min(axis=0) <= 0.25)
    coordinate_similar = np.all(
        (coordinate_values.max(axis=0) - coordinate_values.min(axis=0))
        <= 0.25 * np.maximum(np.abs(coordinate_values.mean(axis=0)), 1e-6)
    )
    overall = length_similar and zero_similar and coordinate_similar
    return (
        f"  Similar sequence lengths: {'yes' if length_similar else 'no'} "
        f"(range {int(lengths.min())}-{int(lengths.max())})\n"
        f"  Similar landmark statistics: {'yes' if overall else 'no'} "
        f"(zero-ratio spread <= 0.25 and coordinate mean/std relative spread <= 25%)"
    )


def main() -> int:
    """Select samples, save plots, and write the diagnostic summary."""
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
    if len(eligible) < SELECTED_CLASS_COUNT:
        raise ValueError("Fewer than three classes have five training samples.")

    rng = np.random.default_rng(SEED)
    selected_classes = sorted(
        rng.choice(eligible, size=SELECTED_CLASS_COUNT, replace=False).tolist()
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "LANDMARK VISUALIZATION SUMMARY",
        f"Seed: {SEED}",
        f"Selected classes: {', '.join(selected_classes)}",
        "Similarity thresholds: sequence length range <= 25% of mean; "
        "zero-ratio spread <= 0.25; coordinate mean/std relative spread <= 25%.",
        "",
    ]

    for class_label in selected_classes:
        class_rows = train[train["label_id"].astype(str) == class_label]
        selected_rows = class_rows.iloc[
            rng.choice(len(class_rows), size=SAMPLES_PER_CLASS, replace=False)
        ]
        summary_lines.append(f"CLASS {class_label}")
        sample_summaries: list[dict[str, float | int]] = []
        for sample_number, (row_index, row) in enumerate(
            selected_rows.iterrows(), start=1
        ):
            sequence = load_sequence(str(row["landmark_path"]), int(row_index))
            plot_path = plot_sample(
                sequence,
                class_label,
                sample_number,
                str(row.get("file_name", row["landmark_path"])),
            )
            statistics = summarize(sequence)
            statistics["landmark_path"] = str(row["landmark_path"])  # type: ignore[assignment]
            sample_summaries.append(statistics)
            summary_lines.extend(
                [
                    f"  Sample {sample_number}: {row['landmark_path']}",
                    f"    Plot: {plot_path}",
                    f"    Sequence length: {statistics['sequence_length']}",
                    f"    Left hand presence/zero ratio: "
                    f"{statistics['left_hand_presence_ratio']:.4f} / "
                    f"{statistics['left_hand_zero_ratio']:.4f}",
                    f"    Right hand presence/zero ratio: "
                    f"{statistics['right_hand_presence_ratio']:.4f} / "
                    f"{statistics['right_hand_zero_ratio']:.4f}",
                    f"    Pose zero ratio: {statistics['pose_zero_ratio']:.4f}",
                    f"    Coordinate min/max: {statistics['coordinate_min']:.6f} / "
                    f"{statistics['coordinate_max']:.6f}",
                    f"    Mean coordinate: {statistics['coordinate_mean']:.6f}",
                    f"    Coordinate standard deviation: {statistics['coordinate_std']:.6f}",
                ]
            )
        summary_lines.append(compare_class_samples(sample_summaries))
        summary_lines.append("")

    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
