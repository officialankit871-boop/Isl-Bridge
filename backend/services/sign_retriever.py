"""Deterministic exact sentence lookup over the M8 sign video index."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


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


def normalize_text(text: str) -> str:
    """Lowercase, trim, collapse whitespace, and remove sentence punctuation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    # Replace punctuation with spaces so forms such as "hello,world" do not
    # accidentally turn into a different word.
    cleaned = text.translate(str.maketrans({char: " " for char in ".,! ?;:" if char != " "}))
    return " ".join(cleaned.lower().split())


class SignRetriever:
    """Load and query the sentence-to-video CSV index in memory."""

    def __init__(self, index_path: str | Path | None = None) -> None:
        self.index_path = Path(index_path) if index_path is not None else (
            Path(__file__).resolve().parents[2] / "data" / "manifests" / "sign_video_index.csv"
        )
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Sign video index does not exist: {self.index_path}")

        self._sentences: dict[str, dict[str, Any]] = {}
        with self.index_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Sign video index is missing required columns: {', '.join(missing)}")
            for line_number, row in enumerate(reader, start=2):
                normalized = normalize_text(row["sentence_normalized"])
                if not normalized:
                    raise ValueError(f"Empty normalized sentence at CSV line {line_number}")
                metadata = {key: row[key] for key in ("sentence", "gloss", "class_name", "label_id")}
                record = self._sentences.get(normalized)
                if record is None:
                    record = {
                        "sentence": metadata["sentence"],
                        "sentence_normalized": normalized,
                        "gloss": metadata["gloss"],
                        "class_name": metadata["class_name"],
                        "label_id": metadata["label_id"],
                        "videos": [],
                        "_video_paths": set(),
                    }
                    self._sentences[normalized] = record
                elif any(record[key] != metadata[key] for key in ("gloss", "class_name", "label_id")):
                    raise ValueError(
                        f"Conflicting sentence metadata for {normalized!r} at CSV line {line_number}"
                    )
                if row["video_path"] not in record["_video_paths"]:
                    record["_video_paths"].add(row["video_path"])
                    record["videos"].append({
                        "video_path": row["video_path"],
                        "relative_video_path": row["relative_video_path"],
                        "file_name": row["file_name"],
                    })

    @staticmethod
    def normalize_text(text: str) -> str:
        return normalize_text(text)

    @property
    def sentence_count(self) -> int:
        return len(self._sentences)

    def find_sentence(self, text: str) -> dict[str, Any]:
        normalized = normalize_text(text)
        record = self._sentences.get(normalized)
        if record is None:
            return {
                "matched": False,
                "sentence": text,
                "sentence_normalized": normalized,
                "gloss": None,
                "class_name": None,
                "label_id": None,
                "video_count": 0,
                "videos": [],
            }
        return {
            "matched": True,
            "sentence": record["sentence"],
            "sentence_normalized": record["sentence_normalized"],
            "gloss": record["gloss"],
            "class_name": record["class_name"],
            "label_id": record["label_id"],
            "video_count": len(record["videos"]),
            "videos": [dict(video) for video in record["videos"]],
        }

    def get_video_variants(self, text: str) -> list[dict[str, str]]:
        return self.find_sentence(text)["videos"]

    def get_by_sentence(self, sentence: str) -> dict[str, Any]:
        return self.find_sentence(sentence)


__all__ = ["SignRetriever", "normalize_text"]
