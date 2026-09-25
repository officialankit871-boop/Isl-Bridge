"""Create reproducible stratified train, validation, and test splits."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("isl_bridge.create_splits")
LANDMARK_MANIFEST = PROJECT_ROOT / "data" / "processed" / "landmarks_manifest.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SUMMARY_PATH = SPLIT_DIR / "split_summary.json"
LABEL_MAP_PATH = PROJECT_ROOT / "data" / "manifests" / "label_map.json"
REQUIRED_COLUMNS = {
    "landmark_path",
    "video_path",
    "relative_video_path",
    "file_name",
    "sentence",
    "class_name",
    "label_id",
    "gloss",
    "num_frames",
    "feature_dim",
    "processing_status",
}
OUTPUT_COLUMNS = [
    "landmark_path",
    "video_path",
    "relative_video_path",
    "file_name",
    "sentence",
    "class_name",
    "label_id",
    "gloss",
    "num_frames",
    "feature_dim",
]
ACCEPTABLE_STATUSES = {"processed", "skipped"}


def parse_args() -> argparse.Namespace:
    """Parse split ratio and random seed options."""
    parser = argparse.ArgumentParser(
        description="Create reproducible train/validation/test CSV splits."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise validation logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate that split ratios are positive and sum to one."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
        raise ValueError("All split ratios must be greater than 0 and less than 1.")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")


def resolve_project_path(value: str) -> Path:
    """Resolve a manifest path relative to the project root when needed."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_text(value: Any) -> str:
    """Normalize text for stable sentence-to-label-map matching."""
    return " ".join(str(value).strip().casefold().split())


def load_label_map() -> dict[str, dict[str, Any]]:
    """Load the existing deterministic label map."""
    if not LABEL_MAP_PATH.is_file():
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")
    try:
        data = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read label map: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Label map must contain an object.")
    return data


def repair_missing_labels(
    frame: pd.DataFrame, label_map: dict[str, dict[str, Any]]
) -> list[str]:
    """Fill blank labels only from a unique sentence in the existing label map."""
    by_sentence = {
        normalize_text(value.get("sentence", "")): (str(label_id), value)
        for label_id, value in label_map.items()
        if value.get("sentence")
    }
    repaired: list[str] = []
    for index, row in frame.iterrows():
        if str(row["label_id"]).strip() and str(row["class_name"]).strip():
            continue
        match = by_sentence.get(normalize_text(row["sentence"]))
        if match is None:
            continue
        label_id, label = match
        frame.at[index, "label_id"] = label_id
        frame.at[index, "class_name"] = str(label.get("class_name", "")).strip()
        repaired.append(str(row["landmark_path"]))
    return repaired


def load_and_validate_manifest() -> tuple[pd.DataFrame, list[str]]:
    """Load the landmark manifest and validate every split-critical invariant."""
    if not LANDMARK_MANIFEST.is_file():
        raise FileNotFoundError(f"Landmark manifest not found: {LANDMARK_MANIFEST}")
    frame = pd.read_csv(LANDMARK_MANIFEST, keep_default_na=False)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Landmark manifest contains no samples.")
    label_map = load_label_map()
    repaired_rows = repair_missing_labels(frame, label_map)

    invalid_statuses = sorted(
        set(frame["processing_status"]) - ACCEPTABLE_STATUSES
    )
    if invalid_statuses:
        raise ValueError(f"Manifest has unacceptable processing statuses: {invalid_statuses}")
    missing_values = {
        column: int(frame[column].astype(str).str.strip().eq("").sum())
        for column in ("landmark_path", "video_path", "label_id", "class_name")
    }
    missing_values = {column: count for column, count in missing_values.items() if count}
    if missing_values:
        raise ValueError(f"Manifest has missing required values: {missing_values}")
    duplicate_landmarks = frame["landmark_path"].duplicated(keep=False)
    duplicate_videos = frame["video_path"].duplicated(keep=False)
    if duplicate_landmarks.any():
        raise ValueError(
            "Duplicate landmark_path values found: "
            f"{sorted(frame.loc[duplicate_landmarks, 'landmark_path'].unique())[:10]}"
        )
    if duplicate_videos.any():
        raise ValueError(
            "Duplicate video_path values found: "
            f"{sorted(frame.loc[duplicate_videos, 'video_path'].unique())[:10]}"
        )
    missing_files = [
        path for path in frame["landmark_path"]
        if not resolve_project_path(path).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} landmark files are missing; first entries: {missing_files[:10]}"
        )
    feature_dims = pd.to_numeric(frame["feature_dim"], errors="coerce")
    invalid_dims = frame.loc[feature_dims.ne(225), "feature_dim"].tolist()
    if invalid_dims:
        raise ValueError(f"Manifest contains feature_dim values other than 225: {invalid_dims[:10]}")
    frame["feature_dim"] = feature_dims.astype(int)
    frame["num_frames"] = pd.to_numeric(frame["num_frames"], errors="raise").astype(int)
    if (frame["num_frames"] <= 0).any():
        raise ValueError("Manifest contains non-positive num_frames values.")
    return (
        frame.sort_values(
            ["label_id", "relative_video_path"], kind="stable"
        ).reset_index(drop=True),
        repaired_rows,
    )


def split_with_sklearn(
    frame: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform the requested two-stage stratified sklearn split."""
    train, temporary = train_test_split(
        frame,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=frame["label_id"],
    )
    temporary_test_ratio = test_ratio / (val_ratio + test_ratio)
    validation, test = train_test_split(
        temporary,
        test_size=temporary_test_ratio,
        random_state=seed,
        stratify=temporary["label_id"],
    )
    return train, validation, test


def fallback_split(
    frame: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Split per class when global stratification is impossible.

    Each class is shuffled deterministically. At least one sample remains in
    training for every class; classes with fewer than three samples cannot
    mathematically occur in all three splits and are reported explicitly.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    constrained_classes: list[str] = []
    for label_id, group in frame.groupby("label_id", sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        sample_count = len(indices)
        if sample_count < 3:
            constrained_classes.append(str(label_id))
        train_count = max(1, round(sample_count * train_ratio))
        remaining = sample_count - train_count
        temporary_ratio = val_ratio / (val_ratio + test_ratio)
        validation_count = round(remaining * temporary_ratio)
        if remaining >= 2:
            validation_count = max(1, validation_count)
        validation_count = min(validation_count, remaining)
        train_parts.append(frame.loc[indices[:train_count]])
        validation_parts.append(frame.loc[indices[train_count:train_count + validation_count]])
        test_parts.append(frame.loc[indices[train_count + validation_count:]])
    return (
        pd.concat(train_parts).sort_index(),
        pd.concat(validation_parts).sort_index(),
        pd.concat(test_parts).sort_index(),
        constrained_classes,
    )


def check_leakage(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Check duplicate video and landmark identities within and across splits."""
    duplicate_videos: set[str] = set()
    duplicate_landmarks: set[str] = set()
    video_locations: dict[str, list[str]] = {}
    landmark_locations: dict[str, list[str]] = {}
    for split_name, frame in splits.items():
        for value in frame["video_path"]:
            video_locations.setdefault(str(value), []).append(split_name)
        for value in frame["landmark_path"]:
            landmark_locations.setdefault(str(value), []).append(split_name)
    for value, locations in video_locations.items():
        if len(locations) > 1:
            duplicate_videos.add(value)
    for value, locations in landmark_locations.items():
        if len(locations) > 1:
            duplicate_landmarks.add(value)
    return {
        "duplicate_videos": sorted(duplicate_videos),
        "duplicate_landmarks": sorted(duplicate_landmarks),
        "passed": not duplicate_videos and not duplicate_landmarks,
    }


def split_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return class counts with deterministic class ordering."""
    return {
        str(label): int(count)
        for label, count in frame["label_id"].value_counts().sort_index().items()
    }


def build_summary(
    splits: dict[str, pd.DataFrame],
    seed: int,
    ratios: dict[str, float],
    warnings: list[str],
    leakage: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON summary for the generated splits."""
    all_data = pd.concat(splits.values(), ignore_index=True)
    return {
        "total_samples": int(len(all_data)),
        "train_samples": int(len(splits["train"])),
        "validation_samples": int(len(splits["val"])),
        "test_samples": int(len(splits["test"])),
        "number_of_classes": int(all_data["label_id"].nunique()),
        "classes_in_each_split": {
            name: sorted(frame["label_id"].unique().tolist())
            for name, frame in splits.items()
        },
        "class_counts": {
            name: split_counts(frame) for name, frame in splits.items()
        },
        "random_seed": seed,
        "split_ratios": ratios,
        "warnings": warnings,
        "leakage_checks": leakage,
    }


def write_outputs(
    splits: dict[str, pd.DataFrame],
    summary: dict[str, Any],
) -> None:
    """Write split CSVs and the JSON summary."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.loc[:, OUTPUT_COLUMNS].sort_values(
            ["label_id", "relative_video_path"], kind="stable"
        ).to_csv(SPLIT_DIR / f"{name}.csv", index=False)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_report(summary: dict[str, Any]) -> None:
    """Print a concise split report."""
    leakage = summary["leakage_checks"]
    print("\n" + "=" * 40)
    print("DATASET SPLIT REPORT")
    print("=" * 40)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Train: {summary['train_samples']}")
    print(f"Validation: {summary['validation_samples']}")
    print(f"Test: {summary['test_samples']}")
    print(f"Total classes: {summary['number_of_classes']}")
    for name in ("train", "val", "test"):
        print(f"{name.title()} classes: {len(summary['classes_in_each_split'][name])}")
    print("\nLeakage check:")
    print(f"Duplicate videos: {len(leakage['duplicate_videos'])}")
    print(f"Duplicate landmarks: {len(leakage['duplicate_landmarks'])}")
    print("\nClass distribution summary:")
    for name, counts in summary["class_counts"].items():
        print(f"{name}: {counts}")
    print(f"\nWarnings: {summary['warnings']}")
    print("=" * 40)


def main() -> int:
    """Validate the landmark manifest, split it, and write reproducible outputs."""
    configure_logging()
    arguments = parse_args()
    try:
        validate_ratios(
            arguments.train_ratio,
            arguments.val_ratio,
            arguments.test_ratio,
        )
        manifest, repaired_rows = load_and_validate_manifest()
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    warnings: list[str] = []
    if repaired_rows:
        warnings.append(
            "Filled missing label_id/class_name from label_map.json for "
            f"{len(repaired_rows)} rows without modifying the source manifest."
        )
    try:
        train, validation, test = split_with_sklearn(
            manifest,
            arguments.seed,
            arguments.train_ratio,
            arguments.val_ratio,
            arguments.test_ratio,
        )
        warnings.append("Used two-stage stratified sklearn split.")
    except ValueError as exc:
        rare_classes = sorted(
            str(label) for label, count in manifest["label_id"].value_counts().items()
            if count < 3
        )
        warnings.append(
            "Normal stratification was mathematically impossible; "
            f"deterministic class-wise fallback used. Cause: {exc}"
        )
        warnings.append(f"Classes with fewer than 3 samples: {rare_classes}")
        train, validation, test, constrained = fallback_split(
            manifest,
            arguments.seed,
            arguments.train_ratio,
            arguments.val_ratio,
            arguments.test_ratio,
        )
        warnings.append(f"Classes unable to appear in all splits: {constrained}")

    splits = {"train": train, "val": validation, "test": test}
    leakage = check_leakage(splits)
    if not leakage["passed"]:
        LOGGER.error("Leakage detected in generated splits.")
        return 1
    summary = build_summary(
        splits,
        arguments.seed,
        {
            "train": arguments.train_ratio,
            "validation": arguments.val_ratio,
            "test": arguments.test_ratio,
        },
        warnings,
        leakage,
    )
    write_outputs(splits, summary)
    print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
