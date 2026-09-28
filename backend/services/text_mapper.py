"""Controlled input mapping to sentences supported by the sign video index."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.services.sign_retriever import SignRetriever, normalize_text


ALIAS_COLUMNS = ("alias", "canonical_sentence")


class TextMapper:
    """Resolve exact indexed sentences first, then explicitly reviewed aliases."""

    def __init__(
        self,
        retriever: SignRetriever | None = None,
        aliases_path: str | Path | None = None,
    ) -> None:
        self.retriever = retriever or SignRetriever()
        self.aliases_path = Path(aliases_path) if aliases_path is not None else (
            Path(__file__).resolve().parents[2] / "data" / "manifests" / "text_aliases.csv"
        )
        self._aliases: dict[str, str] = {}
        if not self.aliases_path.is_file():
            raise FileNotFoundError(f"Text aliases file does not exist: {self.aliases_path}")
        with self.aliases_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = [name for name in ALIAS_COLUMNS if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Text aliases CSV is missing required columns: {', '.join(missing)}")
            for line_number, row in enumerate(reader, start=2):
                alias = normalize_text(row["alias"])
                canonical = normalize_text(row["canonical_sentence"])
                if not alias or not canonical:
                    raise ValueError(f"Empty alias or canonical sentence at CSV line {line_number}")
                if alias in self._aliases:
                    raise ValueError(f"Duplicate normalized alias {alias!r} at CSV line {line_number}")
                indexed = self.retriever.find_sentence(canonical)
                if not indexed["matched"]:
                    raise ValueError(
                        f"Canonical sentence {canonical!r} at CSV line {line_number} is not indexed"
                    )
                self._aliases[alias] = indexed["sentence"]

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def map_text(self, text: str) -> dict[str, Any]:
        """Return the canonical sentence, match kind, and indexed result."""
        normalized = normalize_text(text)
        exact = self.retriever.find_sentence(normalized)
        if exact["matched"]:
            return {"matched": True, "match_type": "exact", "canonical_sentence": exact["sentence"]}
        canonical = self._aliases.get(normalized)
        if canonical is None:
            return {"matched": False, "match_type": "none", "canonical_sentence": text}
        return {"matched": True, "match_type": "alias", "canonical_sentence": canonical}

    def find_sentence(self, text: str) -> dict[str, Any]:
        """Map the input and retrieve its canonical indexed sentence."""
        mapping = self.map_text(text)
        result = self.retriever.find_sentence(mapping["canonical_sentence"])
        result["match_type"] = mapping["match_type"]
        return result


__all__ = ["TextMapper"]
