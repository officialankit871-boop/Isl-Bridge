"""Inspect the ISL-CSLRT sentence-video dataset and create manifests."""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import get_settings  # noqa: E402

LOGGER = logging.getLogger("isl_bridge.dataset_check")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
METADATA_EXTENSIONS = {".csv", ".xlsx", ".txt"}
LABEL_COLUMNS = {"sentence", "sentences", "label", "class", "gloss", "sign_glosses"}
SIGNER_COLUMNS = {"signer", "signer_id", "participant", "subject", "subject_id"}
DEFAULT_FEW_SAMPLE_THRESHOLD = 2
DEFAULT_SHORT_SECONDS = 1.0
DEFAULT_LONG_SECONDS = 300.0


@dataclass
class VideoRecord:
    """Metadata and validation state for one sentence-level video."""

    video_path: str
    relative_video_path: str
    file_name: str
    sentence_folder: str
    sentence: str | None
    class_name: str | None
    label_id: str | None
    gloss: str | None
    signer: str | None
    file_extension: str
    file_size_mb: float
    frame_count: int | None
    fps: float | None
    duration_seconds: float | None
    width: int | None
    height: int | None
    valid: bool
    error: str | None = None


def configure_logging() -> None:
    """Configure logging for the inspection command."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s | %(message)s",
    )


def resolve_dataset_root() -> Path:
    """Resolve the dataset root from ISL_DATASET_ROOT or project configuration."""
    configured = os.getenv("ISL_DATASET_ROOT") or get_settings().dataset_root
    root = Path(configured).expanduser()
    return root if root.is_absolute() else PROJECT_ROOT / root


def discover_files(root: Path, extensions: set[str]) -> list[Path]:
    """Return deterministically sorted files below a root with matching extensions."""
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def normalize_text(value: Any) -> str:
    """Normalize text for metadata matching without changing displayed values."""
    return " ".join(str(value).strip().casefold().split())


def compact_text(value: Any) -> str:
    """Normalize text for conservative whitespace/punctuation typo matching."""
    return "".join(character for character in normalize_text(value) if character.isalnum())


def metadata_for_sentence(
    sentence: str, sentence_metadata: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Resolve sentence metadata exactly or through one unique compact-text match."""
    exact = sentence_metadata.get(normalize_text(sentence))
    if exact and exact.get("gloss"):
        return exact
    candidates = [
        value for key, value in sentence_metadata.items()
        if compact_text(key) == compact_text(sentence)
    ]
    gloss_candidates = [value for value in candidates if value.get("gloss")]
    if len(gloss_candidates) == 1:
        return gloss_candidates[0]
    return exact if exact else (candidates[0] if len(candidates) == 1 else None)


def relative_slash_path(path: Path, root: Path) -> str:
    """Return a portable root-relative path using forward slashes."""
    return path.relative_to(root).as_posix()


def metadata_frame_report(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inspect one CSV/XLSX/TXT file and return records plus a compact report."""
    reports: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        frames = [("CSV", pd.read_csv(path))]
    elif path.suffix.lower() == ".xlsx":
        workbook = pd.ExcelFile(path, engine="openpyxl")
        frames = [(sheet, pd.read_excel(path, sheet_name=sheet, engine="openpyxl"))
                  for sheet in workbook.sheet_names]
    else:
        lines = [
            line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        return [], {
            "filename": path.name,
            "type": "TXT",
            "rows": len(lines),
            "columns": [],
            "sample_records": [{"text": line} for line in lines[:3]],
            "missing_values": {},
        }

    for sheet_name, frame in frames:
        frame = frame.copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        records.extend(frame.fillna("").to_dict(orient="records"))
        reports.append({
            "filename": path.name,
            "sheet_name": sheet_name,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "sample_records": frame.head(3).fillna("").to_dict(orient="records"),
            "missing_values": {
                str(column): int(value) for column, value in frame.isna().sum().items()
            },
        })
    return records, {"sheets": reports}


def inspect_metadata(dataset_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inspect corpus metadata and return reports plus all record-like rows."""
    metadata_dir = dataset_root / "corpus_csv_files"
    metadata_reports: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    metadata_paths = (
        discover_files(metadata_dir, METADATA_EXTENSIONS)
        if metadata_dir.is_dir() else []
    )
    description_path = dataset_root / "ISL_CSLRT.txt"
    if description_path.is_file():
        metadata_paths.append(description_path)
    if not metadata_paths:
        LOGGER.warning("Metadata directory not found: %s", metadata_dir)
        return metadata_reports, all_records
    for path in sorted(metadata_paths):
        try:
            records, report = metadata_frame_report(path)
            all_records.extend(records)
            if "sheets" in report:
                metadata_reports.extend(report["sheets"])
            else:
                metadata_reports.append(report)
        except (OSError, UnicodeError, ValueError, ImportError) as exc:
            LOGGER.warning("Could not inspect metadata file %s: %s", path, exc)
            metadata_reports.append({"filename": path.name, "error": str(exc)})
    return metadata_reports, all_records


def records_by_sentence(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index gloss and optional signer metadata by normalized sentence."""
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        normalized = {
            normalize_text(key).replace(" ", "_"): value for key, value in record.items()
        }
        sentence = next(
            (normalized[key] for key in ("sentence", "sentences", "label", "class")
             if key in normalized and str(normalized[key]).strip()),
            None,
        )
        if sentence is None:
            continue
        key = normalize_text(sentence)
        existing = result.setdefault(
            key, {"sentence": str(sentence).strip(), "gloss": None, "signer": None}
        )
        gloss = next(
            (str(normalized[column]).strip() for column in ("sign_glosses", "gloss")
             if column in normalized and str(normalized[column]).strip()),
            None,
        )
        signer = next(
            (str(normalized[column]).strip() for column in SIGNER_COLUMNS
             if column in normalized and str(normalized[column]).strip()),
            None,
        )
        if gloss:
            existing["gloss"] = gloss
        if signer:
            existing["signer"] = signer
    return result


def build_label_map(sentences: list[str], sentence_metadata: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Create deterministic class IDs from sorted sentence folder names."""
    label_map: dict[str, dict[str, Any]] = {}
    for index, sentence in enumerate(sorted(sentences, key=normalize_text)):
        metadata = metadata_for_sentence(sentence, sentence_metadata) or {}
        class_name = normalize_text(sentence).replace(" ", "_")
        label_map[f"class_{index:03d}"] = {
            "label_id": f"class_{index:03d}",
            "sentence": metadata.get("sentence") or sentence,
            "class_name": class_name,
            "gloss": metadata.get("gloss"),
        }
    return label_map


def inspect_video(
    path: Path,
    dataset_root: Path,
    sentence: str,
    label_id: str | None,
    metadata: dict[str, Any] | None,
    class_name: str | None,
) -> VideoRecord:
    """Inspect one video container without decoding every frame."""
    relative = relative_slash_path(path, dataset_root)
    size_bytes = path.stat().st_size
    record = VideoRecord(
        video_path=path.relative_to(PROJECT_ROOT).as_posix(),
        relative_video_path=relative,
        file_name=path.name,
        sentence_folder=path.parent.name,
        sentence=metadata.get("sentence") if metadata else sentence,
        class_name=class_name,
        label_id=label_id,
        gloss=metadata.get("gloss") if metadata else None,
        signer=metadata.get("signer") if metadata else None,
        file_extension=path.suffix.lower(),
        file_size_mb=round(size_bytes / (1024 * 1024), 6),
        frame_count=None,
        fps=None,
        duration_seconds=None,
        width=None,
        height=None,
        valid=False,
    )
    if size_bytes == 0:
        record.error = "zero_byte_file"
        return record
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            record.error = "unreadable_or_corrupted_video"
            return record
        record.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        record.fps = fps if fps > 0 else None
        record.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        record.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        if record.frame_count > 0 and record.fps:
            record.duration_seconds = record.frame_count / record.fps
        errors = []
        if record.frame_count <= 0:
            errors.append("invalid_frame_count")
        if record.fps is None:
            errors.append("invalid_fps")
        if not record.duration_seconds or record.duration_seconds <= 0:
            errors.append("non_positive_duration")
        if not record.width or not record.height:
            errors.append("invalid_resolution")
        record.error = ",".join(errors) or None
        record.valid = not errors
    except (OSError, ValueError, TypeError) as exc:
        record.error = f"inspection_error:{type(exc).__name__}"
    finally:
        capture.release()
    return record


def find_metadata_video_paths(records: list[dict[str, Any]]) -> set[str]:
    """Extract normalized video suffixes from corpus detail records."""
    paths: set[str] = set()
    for record in records:
        for key, value in record.items():
            key_name = normalize_text(key).replace(" ", "_")
            value_text = normalize_text(str(value).replace("\\", "/"))
            if (
                "file" in key_name
                and "location" in key_name
                and value_text.lower().endswith(tuple(VIDEO_EXTENSIONS))
            ):
                paths.add(value_text)
    return paths


def create_summary(
    records: list[VideoRecord],
    metadata_reports: list[dict[str, Any]],
    label_map: dict[str, dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    dataset_root: Path,
    sentence_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate statistics, quality checks, and metadata availability."""
    valid = [record for record in records if record.valid]
    class_counts = Counter(record.label_id for record in valid if record.label_id)
    durations = [record.duration_seconds for record in valid if record.duration_seconds is not None]
    fps_values = [record.fps for record in valid if record.fps is not None]
    threshold = int(os.getenv("FEW_SAMPLES_THRESHOLD", DEFAULT_FEW_SAMPLE_THRESHOLD))
    short_seconds = float(os.getenv("SHORT_VIDEO_SECONDS", DEFAULT_SHORT_SECONDS))
    long_seconds = float(os.getenv("LONG_VIDEO_SECONDS", DEFAULT_LONG_SECONDS))
    signer_counts = Counter(record.signer for record in valid if record.signer)
    signer_classes: dict[str, set[str]] = defaultdict(set)
    for record in valid:
        if record.signer and record.label_id:
            signer_classes[record.signer].add(record.label_id)
    filename_counts = Counter(record.file_name.casefold() for record in records)
    class_counts_all = Counter(record.label_id for record in records if record.label_id)
    metadata_video_paths = find_metadata_video_paths(metadata_records)
    discovered_paths = {normalize_text(record.relative_video_path) for record in records}
    missing_metadata_files = sorted(
        path for path in metadata_video_paths
        if not any(path.endswith(discovered) or discovered.endswith(path) for discovered in discovered_paths)
    )
    empty_classes = [
        folder.name for folder in (dataset_root / "Videos_Sentence_Level").iterdir()
        if folder.is_dir() and not any(
            child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS
            for child in folder.rglob("*")
        )
    ]
    missing_gloss_classes = sorted(
        label_id for label_id, value in label_map.items() if not value.get("gloss")
    )
    metadata_sentence_keys = set(sentence_metadata)
    folder_sentence_keys = {normalize_text(record.sentence_folder) for record in records}
    unresolved_folder_sentences = sorted(
        sentence for sentence in folder_sentence_keys
        if metadata_for_sentence(sentence, sentence_metadata) is None
    )
    unresolved_metadata_sentences = sorted(
        sentence for sentence in metadata_sentence_keys
        if not any(
            metadata_for_sentence(folder, sentence_metadata)
            and normalize_text(metadata_for_sentence(folder, sentence_metadata)["sentence"]) == sentence
            for folder in folder_sentence_keys
        )
    )
    metadata_mismatches = {
        "video_folders_without_sentence_metadata": unresolved_folder_sentences,
        "metadata_sentences_without_video_folder": unresolved_metadata_sentences,
    }
    dataset_ready = (
        bool(records)
        and len(valid) == len(records)
        and not missing_gloss_classes
        and not metadata_mismatches["video_folders_without_sentence_metadata"]
    )
    fps_statistics = {
        "count": len(fps_values),
        "average": statistics.mean(fps_values) if fps_values else None,
        "minimum": min(fps_values) if fps_values else None,
        "maximum": max(fps_values) if fps_values else None,
        "distribution": dict(Counter(str(value) for value in fps_values)),
    }
    return {
        "dataset_root": dataset_root.relative_to(PROJECT_ROOT).as_posix()
        if dataset_root.is_relative_to(PROJECT_ROOT) else str(dataset_root),
        "total_classes": len(label_map),
        "total_videos": len(records),
        "valid_videos": len(valid),
        "invalid_videos": len(records) - len(valid),
        "class_distribution": dict(sorted(class_counts.items())),
        "class_imbalance": {
            "standard_deviation": (
                statistics.stdev(class_counts_all.values())
                if len(class_counts_all) > 1 else 0
            ),
            "classes_with_very_few_samples": {
                class_id: count for class_id, count in class_counts.items()
                if count <= threshold
            },
        },
        "video_statistics": {
            "minimum_videos_per_class": min(class_counts.values()) if class_counts else 0,
            "maximum_videos_per_class": max(class_counts.values()) if class_counts else 0,
            "mean_videos_per_class": statistics.mean(class_counts.values()) if class_counts else 0,
            "median_videos_per_class": statistics.median(class_counts.values()) if class_counts else 0,
            "average_duration_seconds": statistics.mean(durations) if durations else None,
            "minimum_duration_seconds": min(durations) if durations else None,
            "maximum_duration_seconds": max(durations) if durations else None,
            "average_fps": statistics.mean(fps_values) if fps_values else None,
            "resolution_distribution": dict(Counter(
                f"{record.width}x{record.height}" for record in valid
                if record.width and record.height
            )),
            "extension_distribution": dict(Counter(record.file_extension for record in records)),
        },
        "fps_statistics": fps_statistics,
        "resolution_distribution": dict(Counter(
            f"{record.width}x{record.height}" for record in valid
            if record.width and record.height
        )),
        "signer_information": {
            "available": bool(signer_counts),
            "message": None if signer_counts else "Signer metadata unavailable.",
            "total_signers": len(signer_counts),
            "videos_per_signer": dict(sorted(signer_counts.items())),
            "classes_per_signer": {
                signer: sorted(classes) for signer, classes in sorted(signer_classes.items())
            },
        },
        "metadata_files": metadata_reports,
        "missing_metadata": {
            "classes_without_gloss": missing_gloss_classes,
            "signer": "Signer metadata unavailable.",
        },
        "data_quality_warnings": metadata_mismatches,
        "dataset_ready_for_next_stage": (
            "YES_WITH_WARNINGS" if dataset_ready else "NO"
        ),
        "warnings": {
            "missing_video_files_referenced_by_metadata": missing_metadata_files,
            "unreadable_or_corrupted_videos": [
                record.relative_video_path for record in records
                if record.error and "unreadable_or_corrupted_video" in record.error
            ],
            "empty_class_folders": sorted(empty_classes),
            "duplicate_file_paths": [
                path for path, count in Counter(
                    record.relative_video_path.casefold() for record in records
                ).items() if count > 1
            ],
            "duplicate_filenames": [
                name for name, count in filename_counts.items() if count > 1
            ],
            "zero_byte_files": [
                record.relative_video_path for record in records if record.error == "zero_byte_file"
            ],
            "invalid_fps": [
                record.relative_video_path for record in records
                if record.error and "invalid_fps" in record.error
            ],
            "invalid_frame_count": [
                record.relative_video_path for record in records
                if record.error and "invalid_frame_count" in record.error
            ],
            "non_positive_duration": [
                record.relative_video_path for record in records
                if record.error and "non_positive_duration" in record.error
            ],
            "extremely_short_videos": [
                record.relative_video_path for record in valid
                if record.duration_seconds is not None and record.duration_seconds <= short_seconds
            ],
            "extremely_long_videos": [
                record.relative_video_path for record in valid
                if record.duration_seconds is not None and record.duration_seconds >= long_seconds
            ],
            "classes_with_very_few_samples": {
                class_id: count for class_id, count in class_counts.items() if count <= threshold
            },
            "missing_gloss": missing_gloss_classes,
            "metadata_video_mismatches": metadata_mismatches,
            "missing_label_metadata": [
                record.relative_video_path for record in records if not record.sentence
            ],
        },
    }


def write_outputs(
    records: list[VideoRecord],
    summary: dict[str, Any],
    label_map: dict[str, dict[str, Any]],
) -> None:
    """Write the valid-video manifest, summary, and deterministic label map."""
    output_dir = PROJECT_ROOT / "data" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_fields = [
        "video_path", "relative_video_path", "file_name", "sentence_folder",
        "sentence", "class_name", "label_id", "gloss", "signer", "file_extension", "file_size_mb",
        "frame_count", "fps", "duration_seconds", "width", "height",
    ]
    rows = [
        {field: getattr(record, field) for field in manifest_fields}
        for record in records if record.valid
    ]
    pd.DataFrame(rows, columns=manifest_fields).to_csv(
        output_dir / "manifest.csv", index=False
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_report(summary: dict[str, Any]) -> None:
    """Print a concise human-readable report."""
    stats = summary["video_statistics"]
    print("\n" + "=" * 40)
    print("ISL DATASET REPORT")
    print("=" * 40)
    print(f"Dataset: {summary['dataset_root']}")
    print(f"Sentence classes: {summary['total_classes']}")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Valid videos: {summary['valid_videos']}")
    print(f"Invalid videos: {summary['invalid_videos']}")
    print(f"Average duration: {stats['average_duration_seconds']}")
    print(f"Average FPS: {summary['fps_statistics']['average']}")
    print(f"Resolution: {stats['resolution_distribution']}")
    print(f"Signer metadata: {summary['signer_information']['message'] or 'Available'}")
    print(f"Gloss metadata: {'Available' if not summary['missing_metadata']['classes_without_gloss'] else 'Unavailable for some classes'}")
    print(f"Class imbalance: {summary['warnings']['classes_with_very_few_samples']}")
    print(f"Metadata mismatches: {summary['warnings']['metadata_video_mismatches']}")
    print(f"Metadata files inspected: {len(summary['metadata_files'])}")
    print("Manifest: data/manifests/manifest.csv")
    print("Label map: data/manifests/label_map.json")
    print(f"Warnings: {summary['warnings']}")
    print(f"Dataset ready for landmark extraction: {summary['dataset_ready_for_next_stage']}")
    print("=" * 40)


def main() -> int:
    """Inspect the configured dataset and generate all M2 outputs."""
    configure_logging()
    dataset_root = resolve_dataset_root()
    video_root = dataset_root / "Videos_Sentence_Level"
    if not video_root.is_dir():
        LOGGER.error("Sentence-level video directory not found: %s", video_root)
        return 2
    metadata_reports, metadata_records = inspect_metadata(dataset_root)
    sentence_metadata = records_by_sentence(metadata_records)
    sentence_folders = sorted(
        (path.name for path in video_root.iterdir() if path.is_dir()),
        key=normalize_text,
    )
    label_map = build_label_map(sentence_folders, sentence_metadata)
    ids_by_sentence = {
        normalize_text(value["sentence"]): label_id
        for label_id, value in label_map.items()
    }
    records: list[VideoRecord] = []
    for folder in sorted((path for path in video_root.iterdir() if path.is_dir()), key=lambda p: normalize_text(p.name)):
        sentence_info = metadata_for_sentence(folder.name, sentence_metadata)
        sentence = sentence_info["sentence"] if sentence_info else folder.name
        label_id = ids_by_sentence.get(normalize_text(folder.name))
        for video_path in discover_files(folder, VIDEO_EXTENSIONS):
            records.append(inspect_video(
                video_path, dataset_root, sentence, label_id, sentence_info,
                label_map[label_id]["class_name"] if label_id else None,
            ))
    summary = create_summary(
        records, metadata_reports, label_map, metadata_records, dataset_root,
        sentence_metadata,
    )
    write_outputs(records, summary, label_map)
    print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
