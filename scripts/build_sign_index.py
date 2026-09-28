"""Build and validate the searchable sentence-to-gloss-to-video index."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/manifest.csv"
LABEL_MAP_PATH = PROJECT_ROOT / "data/manifests/label_map.json"
GLOSS_CSV_PATH = (
    PROJECT_ROOT
    / "data/raw/archive/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/corpus_csv_files/ISL Corpus sign glosses.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data/manifests/sign_video_index.csv"
REQUIRED_COLUMNS = (
    "video_path",
    "relative_video_path",
    "file_name",
    "sentence",
    "sentence_normalized",
    "class_name",
    "label_id",
    "gloss",
)


def normalize_sentence(sentence: str) -> str:
    """Lowercase and normalize whitespace, preserving every word and its order."""
    return re.sub(r"\s+", " ", sentence.strip()).lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_video_path(video_path: str, project_root: Path) -> Path:
    """Resolve absolute paths directly and relative paths from the project root."""
    path = Path(video_path)
    return path if path.is_absolute() else project_root / path


def _required_value(row: dict[str, str], name: str, row_number: int) -> str:
    value = row.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Manifest row {row_number} has an empty required field: {name}.")
    return value


def _gloss_lookup(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        sentence = normalize_sentence(row.get("Sentence", ""))
        gloss = row.get("SIGN GLOSSES", "").strip()
        if not sentence or not gloss:
            raise ValueError(f"Gloss CSV row {row_number} has an empty sentence or gloss.")
        previous = lookup.setdefault(sentence, gloss)
        if previous != gloss:
            raise ValueError(f"Gloss CSV has conflicting glosses for sentence: {sentence!r}.")
    return lookup


def build_sign_index(
    manifest_path: Path = MANIFEST_PATH,
    gloss_csv_path: Path = GLOSS_CSV_PATH,
    label_map_path: Path = LABEL_MAP_PATH,
    output_path: Path = OUTPUT_PATH,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, str]]:
    """Build the index in manifest order and validate each source mapping."""
    for description, path in (
        ("Input manifest", manifest_path),
        ("Gloss CSV", gloss_csv_path),
        ("Label map", label_map_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    source_rows = read_csv(manifest_path)
    gloss_lookup = _gloss_lookup(read_csv(gloss_csv_path))
    label_map = json.loads(label_map_path.read_text(encoding="utf-8-sig"))
    if not isinstance(label_map, dict):
        raise ValueError("Label map must be a JSON object keyed by label_id.")

    index: list[dict[str, str]] = []
    video_path_counts: dict[str, int] = defaultdict(int)
    for row_number, source in enumerate(source_rows, start=2):
        output_row = {
            name: _required_value(source, name, row_number)
            for name in ("video_path", "relative_video_path", "file_name", "sentence", "class_name", "label_id", "gloss")
        }
        output_row["sentence_normalized"] = normalize_sentence(output_row["sentence"])
        expected_gloss = gloss_lookup.get(output_row["sentence_normalized"])
        if expected_gloss is None:
            raise ValueError(
                f"Manifest row {row_number} sentence has no exact gloss CSV match: "
                f"{output_row['sentence']!r}."
            )
        if output_row["gloss"].strip() != expected_gloss:
            raise ValueError(
                f"Gloss mismatch for {output_row['sentence']!r}: manifest has "
                f"{output_row['gloss']!r}, gloss CSV has {expected_gloss!r}."
            )

        label = label_map.get(output_row["label_id"])
        if (
            not isinstance(label, dict)
            or label.get("label_id") != output_row["label_id"]
            or label.get("class_name") != output_row["class_name"]
        ):
            raise ValueError(
                f"Label-map mismatch for {output_row['label_id']!r}/"
                f"{output_row['class_name']!r}."
            )

        video_path_counts[output_row["video_path"]] += 1
        index.append(output_row)

    duplicate_paths = [path for path, count in video_path_counts.items() if count != 1]
    if duplicate_paths:
        raise ValueError(f"Manifest video_path values are not unique: {duplicate_paths[:5]}")
    if len(index) != len(source_rows):
        raise ValueError("Index row count does not match the input manifest.")

    missing_files = [
        row["video_path"]
        for row in index
        if not resolve_video_path(row["video_path"], project_root).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} manifest video_path values do not resolve from "
            f"project root {project_root}: {missing_files[:10]}"
        )

    collision: dict[str, set[str]] = defaultdict(set)
    sentence_video_counts: dict[str, int] = defaultdict(int)
    for row in index:
        collision[row["gloss"].strip()].add(row["sentence_normalized"])
        sentence_video_counts[row["sentence_normalized"]] += 1
    collisions = {gloss: sentences for gloss, sentences in collision.items() if len(sentences) > 1}
    expected_collision = {
        "WHAT YOU DO": {"what are you doing", "what do you do"}
    }
    if any(collisions.get(gloss) != sentences for gloss, sentences in expected_collision.items()):
        raise ValueError(f"Expected shared gloss collision is missing or changed: {collisions}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(index)

    # Check the serialized artifact as well as the in-memory records.
    serialized = read_csv(output_path)
    if len(serialized) != len(source_rows):
        raise ValueError("Written index row count does not match the input manifest.")
    if serialized and tuple(serialized[0].keys()) != REQUIRED_COLUMNS:
        raise ValueError("Written index columns do not match the required schema.")
    if source_rows and not serialized:
        raise ValueError("Written index columns do not match the required schema.")
    serialized_paths = [row["video_path"] for row in serialized]
    source_paths = [row["video_path"] for row in index]
    if serialized_paths != source_paths:
        raise ValueError("Written index video_path values do not match the manifest order.")
    if len(set(serialized_paths)) != len(serialized_paths):
        raise ValueError("Written index contains duplicate video_path values.")
    if any(not all(row.get(column, "").strip() for column in REQUIRED_COLUMNS) for row in serialized):
        raise ValueError("Written index contains empty required values.")

    unique_sentences = {row["sentence_normalized"] for row in serialized}
    unique_glosses = {row["gloss"] for row in serialized}
    print("M8.2 sign video index build")
    print(f"Manifest rows: {len(source_rows)}")
    print(f"Index rows: {len(serialized)}")
    print(f"Unique video paths: {len(set(serialized_paths))}")
    print(f"Unique sentences: {len(unique_sentences)}")
    print(f"Unique glosses: {len(unique_glosses)}")
    print("Missing required values: 0")
    print("Duplicate video paths: 0")
    print("Missing video files: 0")
    print("Gloss mismatches: 0")
    print("Label-map mismatches: 0")
    print(f"Shared gloss collisions: {len(collisions)}")
    for gloss, sentences in sorted(collisions.items()):
        print(f"  {gloss}: {' | '.join(sorted(sentences))}")
    print(f"Video variants preserved: {len(index)} rows across {len(sentence_video_counts)} sentences")
    print(f"Index written: {output_path}")
    return serialized


def main() -> int:
    build_sign_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
