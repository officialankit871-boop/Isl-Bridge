"""Tests for exact sentence retrieval from the checked-in video index."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from backend.services.sign_retriever import REQUIRED_COLUMNS, SignRetriever, normalize_text


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "manifests" / "sign_video_index.csv"


class TestSignRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = SignRetriever(INDEX)

    def test_initializes_and_indexes_101_sentences(self) -> None:
        self.assertEqual(self.retriever.sentence_count, 101)

    def test_normalization(self) -> None:
        self.assertEqual(normalize_text("  Are   You Free Today? "), "are you free today")
        self.assertEqual(normalize_text("Hello,world!"), "hello world")

    def test_case_whitespace_and_punctuation_lookup(self) -> None:
        for query in (
            "Are you free today",
            "are you free today",
            "  ARE   YOU   FREE   TODAY  ",
            "Are you free today?",
        ):
            with self.subTest(query=query):
                result = self.retriever.find_sentence(query)
                self.assertTrue(result["matched"])
                self.assertEqual(result["sentence_normalized"], "are you free today")
                self.assertEqual(result["sentence"], "are you free today")

    def test_variants_and_required_metadata(self) -> None:
        result = self.retriever.find_sentence("Are you free today?")
        self.assertEqual(result["video_count"], len(result["videos"]))
        self.assertGreater(result["video_count"], 1)
        paths = [video["video_path"] for video in result["videos"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all({"video_path", "relative_video_path", "file_name"} <= v.keys() for v in result["videos"]))
        for field in ("sentence", "sentence_normalized", "gloss", "class_name", "label_id"):
            self.assertTrue(result[field], field)
        self.assertEqual(self.retriever.get_video_variants("are you free today"), result["videos"])
        self.assertEqual(self.retriever.get_by_sentence("are you free today"), result)

    def test_unknown_sentence(self) -> None:
        result = self.retriever.find_sentence("This sentence does not exist in the ISL dataset")
        self.assertFalse(result["matched"])
        self.assertEqual(result["video_count"], 0)
        self.assertEqual(result["videos"], [])
        self.assertIsNone(result["gloss"])
        self.assertIsNone(result["class_name"])
        self.assertIsNone(result["label_id"])

    def test_shared_gloss_does_not_drive_lookup(self) -> None:
        doing = self.retriever.find_sentence("what are you doing")
        do = self.retriever.find_sentence("what do you do")
        self.assertTrue(doing["matched"])
        self.assertTrue(do["matched"])
        self.assertEqual(doing["sentence"], "what are you doing")
        self.assertEqual(do["sentence"], "what do you do")
        self.assertEqual(doing["gloss"], "WHAT YOU DO")
        self.assertEqual(do["gloss"], "WHAT YOU DO")

    def test_missing_file_and_columns_fail_clearly(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            SignRetriever(INDEX.parent / "missing.csv")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("sentence,gloss\na,b\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required columns"):
                SignRetriever(path)

    def test_conflicting_duplicate_sentence_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflict.csv"
            rows = [
                dict.fromkeys(REQUIRED_COLUMNS, ""),
                dict.fromkeys(REQUIRED_COLUMNS, ""),
            ]
            for row in rows:
                row.update({"sentence": "hello there", "sentence_normalized": "hello there", "class_name": "hello", "label_id": "class_1", "video_path": "a.mp4", "relative_video_path": "a.mp4", "file_name": "a.mp4"})
            rows[0]["gloss"] = "HELLO"
            rows[1]["gloss"] = "THERE"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Conflicting sentence metadata"):
                SignRetriever(path)


if __name__ == "__main__":
    unittest.main()
