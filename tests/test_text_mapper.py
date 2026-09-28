"""Tests for exact and curated sentence mapping."""

import csv

import pytest

from backend.services.sign_retriever import SignRetriever
from backend.services.text_mapper import TextMapper


def _write_aliases(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("alias", "canonical_sentence"))
        writer.writerows(rows)


def test_exact_and_normalized_sentences_are_exact(tmp_path):
    mapper = TextMapper(SignRetriever(), tmp_path / "empty.csv") if False else None
    aliases = tmp_path / "aliases.csv"
    _write_aliases(aliases, [])
    mapper = TextMapper(SignRetriever(), aliases)
    assert mapper.find_sentence("  ARE   YOU FREE TODAY? ")["match_type"] == "exact"


def test_valid_alias_maps_to_an_indexed_canonical_sentence(tmp_path):
    aliases = tmp_path / "aliases.csv"
    _write_aliases(aliases, [("verified alternate", "are you free today")])
    mapper = TextMapper(SignRetriever(), aliases)
    result = mapper.find_sentence("VERIFIED alternate!")
    assert result["matched"] is True
    assert result["match_type"] == "alias"
    assert result["sentence"] == "are you free today"
    assert result["video_count"] > 0


def test_unknown_input_has_no_match(tmp_path):
    aliases = tmp_path / "aliases.csv"
    _write_aliases(aliases, [])
    result = TextMapper(SignRetriever(), aliases).find_sentence("unknown phrase")
    assert result["matched"] is False
    assert result["match_type"] == "none"
    assert result["videos"] == []


def test_duplicate_normalized_aliases_are_rejected(tmp_path):
    aliases = tmp_path / "aliases.csv"
    _write_aliases(aliases, [("hello!", "are you free today"), (" HELLO ", "what are you doing")])
    with pytest.raises(ValueError, match="Duplicate normalized alias"):
        TextMapper(SignRetriever(), aliases)


def test_canonical_sentences_are_validated_against_index(tmp_path):
    aliases = tmp_path / "aliases.csv"
    _write_aliases(aliases, [("alternate", "made up sentence")])
    with pytest.raises(ValueError, match="is not indexed"):
        TextMapper(SignRetriever(), aliases)


def test_production_aliases_are_unique_and_indexed():
    mapper = TextMapper()
    assert len(mapper.aliases) == len(set(mapper.aliases))
    assert all(mapper.retriever.find_sentence(sentence)["matched"] for sentence in mapper.aliases.values())
