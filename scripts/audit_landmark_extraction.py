"""Audit landmark extraction semantics and inspect deterministic training samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = PROJECT_ROOT / "scripts" / "extract_landmarks.py"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
PROCESSING_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "landmarks_manifest.csv"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "landmark_extraction_audit.json"

FEATURE_DIMENSION = 225
LANDMARK_COUNT = 75
SAMPLE_SIZE = 20
SEED = 42
NEAR_ZERO_NORM_THRESHOLD = 1e-6
COMPONENTS: dict[str, tuple[int, int]] = {
    "pose": (0, 33),
    "left_hand": (33, 54),
    "right_hand": (54, 75),
}


def resolve_path(value: str) -> Path:
    """Resolve project-relative landmark paths."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def source_definition_excerpt(source_lines: list[str], name: str) -> dict[str, Any]:
    """Return line-numbered source for a top-level function or class."""
    start_index = next(
        (
            index
            for index, line in enumerate(source_lines)
            if (
                line.startswith(f"def {name}(")
                or line.startswith(f"class {name}:")
            )
        ),
        None,
    )
    if start_index is None:
        return {"found": False, "line_start": None, "lines": []}

    end_index = len(source_lines)
    for index in range(start_index + 1, len(source_lines)):
        line = source_lines[index]
        if line.startswith(("def ", "class ")):
            end_index = index
            break
    excerpt_lines = source_lines[start_index:end_index]
    return {
        "found": True,
        "line_start": start_index + 1,
        "lines": [
            {"line": start_index + offset + 1, "text": line}
            for offset, line in enumerate(excerpt_lines)
        ],
    }


def summarize_component(values: np.ndarray) -> dict[str, Any]:
    """Summarize a component's stored coordinates and landmark vectors."""
    finite_coordinates = values[np.isfinite(values)]
    landmark_vectors = values.reshape(values.shape[0], -1, 3)
    finite_landmarks = np.isfinite(landmark_vectors).all(axis=2)
    exact_zero_landmarks = finite_landmarks & np.all(
        landmark_vectors == 0.0,
        axis=2,
    )
    landmark_magnitudes = np.linalg.norm(
        np.where(np.isfinite(landmark_vectors), landmark_vectors, 0.0),
        axis=2,
    )
    near_zero_landmarks = (
        finite_landmarks & (landmark_magnitudes <= NEAR_ZERO_NORM_THRESHOLD)
    )

    if finite_coordinates.size:
        flattened_finite = values.reshape(-1, 3)
        axis_ranges = {
            axis: {
                "min": float(np.min(flattened_finite[np.isfinite(flattened_finite[:, index]), index])),
                "max": float(np.max(flattened_finite[np.isfinite(flattened_finite[:, index]), index])),
            }
            for index, axis in enumerate(("x", "y", "z"))
            if np.isfinite(flattened_finite[:, index]).any()
        }
    else:
        axis_ranges = {}

    return {
        "shape": list(values.shape),
        "coordinate_count": int(values.size),
        "finite_coordinate_count": int(finite_coordinates.size),
        "exact_zero_coordinate_count": int(np.sum(values == 0.0)),
        "exact_zero_landmark_count": int(exact_zero_landmarks.sum()),
        "near_zero_landmark_count": int(near_zero_landmarks.sum()),
        "near_zero_landmark_norm_threshold": NEAR_ZERO_NORM_THRESHOLD,
        "coordinate_ranges": axis_ranges,
        "mean_landmark_vector_magnitude": (
            float(np.mean(landmark_magnitudes[finite_landmarks]))
            if finite_landmarks.any()
            else None
        ),
    }


def per_frame_hand_magnitudes(values: np.ndarray) -> dict[str, list[float | None]]:
    """Return per-frame L2 magnitudes for left, right, and combined hands."""
    output: dict[str, list[float | None]] = {}
    for name, (start, end) in (
        ("left_hand", COMPONENTS["left_hand"]),
        ("right_hand", COMPONENTS["right_hand"]),
        ("both_hands", (33, 75)),
    ):
        component = values[:, start:end, :]
        rows: list[float | None] = []
        for frame in component:
            if np.isfinite(frame).all():
                rows.append(float(np.linalg.norm(frame.reshape(-1))))
            else:
                rows.append(None)
        output[name] = rows
    return output


def inspect_sample(
    split_row_index: int,
    row: pd.Series,
) -> dict[str, Any]:
    """Load and inspect one sampled training file without changing it."""
    raw_path = str(row["landmark_path"]).strip()
    path = resolve_path(raw_path)
    sample: dict[str, Any] = {
        "training_split_row": split_row_index + 2,
        "class_name": str(row.get("class_name", "")),
        "class_id": str(row.get("label_id", "")),
        "sentence": str(row.get("sentence", "")),
        "landmark_path": str(path),
        "loaded": False,
        "shape": None,
        "dtype": None,
        "shape_valid": False,
        "components": {},
        "per_frame_hand_magnitudes": {},
        "nan_inf_frame_count": None,
        "error": None,
    }
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError, EOFError) as exc:
        sample["error"] = f"Could not load file: {exc}"
        return sample

    sample["shape"] = list(values.shape)
    sample["dtype"] = str(values.dtype)
    sample["shape_valid"] = bool(
        values.ndim == 2
        and values.shape[0] > 0
        and values.shape[1] == FEATURE_DIMENSION
    )
    if not sample["shape_valid"]:
        sample["error"] = (
            f"Expected a non-empty (T, {FEATURE_DIMENSION}) array; "
            f"found {values.shape}."
        )
        return sample
    if not np.issubdtype(values.dtype, np.number):
        sample["error"] = f"Expected numeric dtype; found {values.dtype}."
        return sample

    sample["loaded"] = True
    reshaped = np.asarray(values).reshape(-1, LANDMARK_COUNT, 3)
    sample["nan_inf_frame_count"] = int(
        (~np.isfinite(values).all(axis=1)).sum()
    )
    for name, (landmark_start, landmark_end) in COMPONENTS.items():
        sample["components"][name] = summarize_component(
            reshaped[:, landmark_start:landmark_end, :]
        )
    sample["components"]["both_hands"] = summarize_component(
        reshaped[:, 33:75, :]
    )
    sample["per_frame_hand_magnitudes"] = per_frame_hand_magnitudes(reshaped)
    return sample


def inspect_processing_manifest(
    path: Path,
    sampled_paths: set[str],
) -> dict[str, Any]:
    """Check if existing sidecar extraction counters preserve useful detections."""
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "rows": 0,
        "matching_sample_rows": 0,
        "processing_status_counts": {},
        "matching_rows": [],
        "usable_detection_counter_rows": 0,
        "note": None,
    }
    if not path.is_file():
        result["note"] = "No processing manifest was found."
        return result

    manifest = pd.read_csv(path, keep_default_na=False)
    result["rows"] = int(len(manifest))
    if "landmark_path" not in manifest.columns:
        result["note"] = "Processing manifest has no landmark_path column."
        return result
    if "processing_status" in manifest:
        result["processing_status_counts"] = {
            str(key): int(value)
            for key, value in manifest["processing_status"].value_counts().items()
        }

    counters = (
        "frames_with_pose",
        "frames_with_left_hand",
        "frames_with_right_hand",
    )
    records: list[dict[str, Any]] = []
    normalized_target_paths = {
        str(Path(value)).replace("\\", "/").casefold() for value in sampled_paths
    }
    for _, row in manifest.iterrows():
        manifest_path = str(row["landmark_path"]).replace("\\", "/").casefold()
        if manifest_path not in normalized_target_paths:
            continue
        record = {
            "landmark_path": str(row["landmark_path"]),
            "processing_status": str(row.get("processing_status", "")),
        }
        record.update(
            {
                counter: (
                    int(row[counter])
                    if counter in manifest.columns and str(row[counter]).strip()
                    else None
                )
                for counter in counters
            }
        )
        records.append(record)
    result["matching_sample_rows"] = len(records)
    result["matching_rows"] = records
    result["usable_detection_counter_rows"] = sum(
        record["processing_status"] == "processed"
        and any(record[counter] is not None for counter in counters)
        for record in records
    )
    if result["rows"] and result["processing_status_counts"].get("skipped", 0) == result["rows"]:
        result["note"] = (
            "Every row is marked skipped; process_row returns zero detection counters "
            "for skipped files, so these counters do not recover per-frame presence."
        )
    else:
        result["note"] = (
            "Sidecar counters are aggregate counts for processed files only; they are "
            "not per-frame presence flags."
        )
    return result


def extraction_findings(source_excerpts: dict[str, Any]) -> dict[str, Any]:
    """Record conclusions established from the extraction implementation."""
    return {
        "mediapipe_outputs": {
            "pose": (
                "result.pose_landmarks is passed to extract_group(result, 33); "
                "when available, each landmark contributes x, y, z."
            ),
            "left_hand": (
                "result.left_hand_landmarks is passed to extract_group(result, 21); "
                "when available, each landmark contributes x, y, z."
            ),
            "right_hand": (
                "result.right_hand_landmarks is passed to extract_group(result, 21); "
                "when available, each landmark contributes x, y, z."
            ),
        },
        "missing_group_handling": (
            "empty_landmark_group creates a float32 all-zero (count, 3) array. "
            "extract_group returns it when the MediaPipe result is None, has no "
            "landmark entries, or has an unexpected coordinate shape."
        ),
        "normalization_order": (
            "Pose and hand groups are extracted/zero-filled, concatenated, then "
            "BodyNormalizer.normalize is applied to the complete 75-landmark array "
            "for each frame before the frame is flattened and saved."
        ),
        "zero_fill_normalization_effect": (
            "For a missing group, each zero coordinate is transformed by "
            "(0 - center) / scale when a valid shoulder reference has been seen. "
            "Thus missing-group zeros generally become repeated -center/scale "
            "coordinates. If no reference has been established, center=0 and "
            "scale=1, so zeros remain zero. nan_to_num then converts non-finite "
            "normalized outputs to zero."
        ),
        "presence_metadata": (
            "extract_video counts pose/left/right detections during processing. "
            "The processing manifest stores aggregate frames_with_* counts, not "
            "a per-frame mask. process_row sets these counters to zero for files "
            "reused with processing_status='skipped'."
        ),
        "current_npy_recoverability": (
            "The .npy file contains only normalized x/y/z values. It does not store "
            "presence flags, visibility, or the center/scale history, so the original "
            "detector-presence state cannot be reliably reconstructed."
        ),
        "source_excerpt_line_starts": {
            name: excerpt["line_start"]
            for name, excerpt in source_excerpts.items()
        },
    }


def print_sample(sample: dict[str, Any]) -> None:
    """Print a compact summary of one inspected .npy file."""
    print(
        f"\nTrain row {sample['training_split_row']} | "
        f"{sample['class_id']} | {sample['class_name']} | "
        f"{sample['landmark_path']}"
    )
    if not sample["loaded"]:
        print(f"  INVALID: {sample['error']}")
        return
    print(
        f"  shape={tuple(sample['shape'])} dtype={sample['dtype']} "
        f"shape_valid={sample['shape_valid']} "
        f"NaN/Inf frames={sample['nan_inf_frame_count']}"
    )
    for name, stats in sample["components"].items():
        ranges = stats["coordinate_ranges"]
        print(
            f"  {name}: ranges={ranges} exact_zero_landmarks="
            f"{stats['exact_zero_landmark_count']} "
            f"near_zero_landmarks={stats['near_zero_landmark_count']} "
            f"mean_landmark_magnitude={stats['mean_landmark_vector_magnitude']}"
        )
    magnitudes = sample["per_frame_hand_magnitudes"]
    for name in ("left_hand", "right_hand", "both_hands"):
        finite = [value for value in magnitudes[name] if value is not None]
        print(
            f"  {name} per-frame magnitude: count={len(finite)} "
            f"mean={float(np.mean(finite)) if finite else None} "
            f"min={float(np.min(finite)) if finite else None} "
            f"max={float(np.max(finite)) if finite else None}"
        )


def main() -> int:
    """Inspect extraction source and a deterministic sample of training files."""
    np.random.seed(SEED)
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Landmark extraction source not found: {SOURCE_PATH}")
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Training split not found: {TRAIN_PATH}")

    source_lines = SOURCE_PATH.read_text(encoding="utf-8").splitlines()
    required_definitions = (
        "empty_landmark_group",
        "extract_group",
        "BodyNormalizer",
        "extract_video",
        "process_row",
    )
    source_excerpts = {
        name: source_definition_excerpt(source_lines, name)
        for name in required_definitions
    }
    missing_definitions = [
        name for name, excerpt in source_excerpts.items() if not excerpt["found"]
    ]
    if missing_definitions:
        raise ValueError(
            "Extraction implementation changed; audit definitions not found: "
            + ", ".join(missing_definitions)
        )

    train_frame = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    required_columns = {"landmark_path", "class_name"}
    missing_columns = required_columns - set(train_frame.columns)
    if missing_columns:
        raise ValueError(
            f"{TRAIN_PATH} is missing required columns: {sorted(missing_columns)}"
        )
    sample_count = min(SAMPLE_SIZE, len(train_frame))
    if sample_count == 0:
        raise ValueError(f"No training samples found in {TRAIN_PATH}.")
    rng = np.random.default_rng(SEED)
    selected_indices = sorted(
        int(index)
        for index in rng.choice(len(train_frame), size=sample_count, replace=False)
    )
    samples = [
        inspect_sample(index, train_frame.iloc[index])
        for index in selected_indices
    ]
    sampled_paths = {sample["landmark_path"] for sample in samples}
    processing_manifest = inspect_processing_manifest(
        PROCESSING_MANIFEST_PATH,
        sampled_paths,
    )
    findings = extraction_findings(source_excerpts)

    loaded_samples = [sample for sample in samples if sample["loaded"]]
    hand_exact_zero_counts = {
        hand: sum(
            sample["components"][hand]["exact_zero_landmark_count"]
            for sample in loaded_samples
        )
        for hand in ("left_hand", "right_hand")
    }
    report = {
        "audit": {
            "seed": SEED,
            "training_split": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
            "requested_sample_count": SAMPLE_SIZE,
            "selected_sample_count": sample_count,
            "selected_training_row_indices_zero_based": selected_indices,
            "analyzed_files": len(loaded_samples),
            "invalid_sample_files": len(samples) - len(loaded_samples),
            "near_zero_landmark_norm_threshold": NEAR_ZERO_NORM_THRESHOLD,
            "near_zero_definition": (
                "Euclidean norm of one landmark's x/y/z vector <= 1e-6."
            ),
            "hand_exact_zero_landmark_counts_across_sample": hand_exact_zero_counts,
        },
        "extraction_source": {
            "path": str(SOURCE_PATH.relative_to(PROJECT_ROOT)),
            "findings": findings,
            "evidence": source_excerpts,
        },
        "processing_manifest": processing_manifest,
        "sample_inspection": samples,
        "conclusions": {
            "normalization_order": findings["normalization_order"],
            "missing_landmark_handling": findings["missing_group_handling"],
            "normalization_changes_zero_filled_coordinates": True,
            "hand_presence_information_retained": False,
            "current_npy_files_reliably_measure_detector_coverage": False,
            "reextraction_recommended_for_detector_coverage_diagnostic": True,
            "reextraction_metadata_recommendation": [
                "Save left_hand_present per frame.",
                "Save right_hand_present per frame.",
                "Save pose_present per frame.",
                "Save per-frame visibility/presence/confidence values if supplied by the landmark API.",
                "Store presence metadata alongside each landmark sequence without inferring it from normalized coordinates.",
                "If normalization must be reversible, record the per-frame body-normalization center and scale.",
            ],
        },
    }

    print("============================================================")
    print("LANDMARK EXTRACTION IMPLEMENTATION AUDIT")
    print("============================================================")
    print(f"Source: {SOURCE_PATH.relative_to(PROJECT_ROOT)}")
    print(
        "MediaPipe access: pose_landmarks, left_hand_landmarks, and "
        "right_hand_landmarks are converted to x/y/z arrays when present."
    )
    print(
        "Missing group: empty_landmark_group returns an all-zero float32 array; "
        "this is concatenated with other groups before normalization."
    )
    print(
        "Normalization: the full concatenated pose+hands frame is normalized "
        "after zero-filling using (landmarks - center) / scale."
    )
    print(
        "Effect: when a shoulder reference exists, a zero-filled missing hand is "
        "shifted to -center/scale, so exact-zero tests no longer identify it."
    )
    print(
        "Saved presence metadata: aggregate detection counters exist for freshly "
        "processed rows, but per-frame flags are not stored; skipped rows get zero counters."
    )
    print(
        f"\nSelected {len(samples)} training videos deterministically with seed={SEED}."
    )
    for sample in samples:
        print_sample(sample)

    print("\nPROCESSING MANIFEST AUDIT")
    print(f"Exists: {processing_manifest['exists']}")
    print(f"Rows: {processing_manifest['rows']}")
    print(f"Statuses: {processing_manifest['processing_status_counts']}")
    print(f"Usable sampled aggregate counter rows: {processing_manifest['usable_detection_counter_rows']}")
    print(processing_manifest["note"])

    print("\nLANDMARK EXTRACTION AUDIT")
    print("Normalization order: zero-fill components, concatenate, then normalize whole frame.")
    print("Missing-landmark handling: all-zero group before normalization; transformed by body center/scale.")
    print("Hand-presence information retained: NO (not per frame in .npy; sidecar counters are aggregate).")
    print("Current .npy files can reliably measure detector coverage: NO.")
    print("Re-extraction recommended for this diagnostic: YES, if detector coverage is required.")
    print(
        "Future extraction should save per-frame left_hand_present, right_hand_present, "
        "pose_present, and visibility/presence values if available; optionally save "
        "normalization center and scale."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nAudit report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
