"""Focused tests for the lightweight sentence-to-video index builder."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_sign_index import REQUIRED_COLUMNS, build_sign_index, normalize_sentence


class TestSignVideoIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest_path = self.root / "manifest.csv"
        self.gloss_path = self.root / "gloss.csv"
        self.label_map_path = self.root / "label_map.json"
        self.output_path = self.root / "index.csv"

        self.rows = [
            {
                "video_path": "videos/doing-a.mp4",
                "relative_video_path": "doing-a.mp4",
                "file_name": "doing-a.mp4",
                "sentence": " WHAT   ARE\tYOU DOING ",
                "class_name": "what_are_you_doing",
                "label_id": "class_001",
                "gloss": "WHAT YOU DO",
            },
            {
                "video_path": "videos/doing-b.mp4",
                "relative_video_path": "doing-b.mp4",
                "file_name": "doing-b.mp4",
                "sentence": " WHAT   ARE\tYOU DOING ",
                "class_name": "what_are_you_doing",
                "label_id": "class_001",
                "gloss": "WHAT YOU DO",
            },
            {
                "video_path": "videos/do-you-do.mp4",
                "relative_video_path": "do-you-do.mp4",
                "file_name": "do-you-do.mp4",
                "sentence": "what do you do",
                "class_name": "what_do_you_do",
                "label_id": "class_002",
                "gloss": "WHAT YOU DO",
            },
        ]
        for row in self.rows:
            video = self.root / row["video_path"]
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"fixture")

        with self.manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)
        with self.gloss_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("Sentence", "SIGN GLOSSES"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"Sentence": "what are you doing", "SIGN GLOSSES": "WHAT YOU DO"},
                    {"Sentence": "what do you do", "SIGN GLOSSES": "WHAT YOU DO"},
                ]
            )
        self.label_map_path.write_text(
            json.dumps(
                {
                    "class_001": {
                        "label_id": "class_001",
                        "sentence": "what are you doing",
                        "class_name": "what_are_you_doing",
                        "gloss": "WHAT YOU DO",
                    },
                    "class_002": {
                        "label_id": "class_002",
                        "sentence": "what do you do",
                        "class_name": "what_do_you_do",
                        "gloss": "WHAT YOU DO",
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build(self) -> list[dict[str, str]]:
        return build_sign_index(
            manifest_path=self.manifest_path,
            gloss_csv_path=self.gloss_path,
            label_map_path=self.label_map_path,
            output_path=self.output_path,
            project_root=self.root,
        )

    def test_schema_count_required_values_and_unique_paths(self) -> None:
        result = self.build()

        self.assertEqual(len(result), len(self.rows))
        self.assertEqual(tuple(result[0]), REQUIRED_COLUMNS)
        self.assertEqual(len({row["video_path"] for row in result}), len(result))
        self.assertTrue(all(row[column].strip() for row in result for column in REQUIRED_COLUMNS))

    def test_sentence_normalization_is_conservative(self) -> None:
        result = self.build()
        self.assertEqual(result[0]["sentence_normalized"], "what are you doing")
        self.assertEqual(
            normalize_sentence("  KEEP every word\t in order "),
            "keep every word in order",
        )

    def test_shared_gloss_and_multiple_video_variants_are_preserved(self) -> None:
        result = self.build()
        doing = [row for row in result if row["sentence_normalized"] == "what are you doing"]
        shared = {
            row["sentence_normalized"]
            for row in result
            if row["gloss"] == "WHAT YOU DO"
        }

        self.assertEqual(len(doing), 2)
        self.assertEqual({row["video_path"] for row in doing}, {"videos/doing-a.mp4", "videos/doing-b.mp4"})
        self.assertEqual(shared, {"what are you doing", "what do you do"})

    def test_output_is_deterministic(self) -> None:
        self.build()
        first = self.output_path.read_bytes()
        self.build()

        self.assertEqual(self.output_path.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
