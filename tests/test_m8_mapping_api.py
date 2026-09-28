"""M8.6 mapping behavior at the sign API boundary."""

from fastapi.testclient import TestClient

from backend.api.routes_sign import get_sign_retriever
from backend.main import app
from backend.services.text_mapper import TextMapper


client = TestClient(app)


def test_exact_response_marks_exact_match():
    result = client.post("/sign/translate", json={"text": "  ARE YOU FREE TODAY? "}).json()
    assert result["matched"] is True
    assert result["match_type"] == "exact"
    assert result["sentence"] == "are you free today"


def test_alias_response_retrieves_video_variants(tmp_path):
    aliases = tmp_path / "aliases.csv"
    aliases.write_text("alias,canonical_sentence\nverified alternate,are you free today\n", encoding="utf-8")
    app.state.sign_retriever = get_sign_retriever()
    app.state.text_mapper = TextMapper(app.state.sign_retriever, aliases)
    try:
        result = client.post("/sign/translate", json={"text": "verified alternate"}).json()
        assert result["matched"] is True
        assert result["match_type"] == "alias"
        assert result["sentence"] == "are you free today"
        assert result["videos"]
    finally:
        app.state.text_mapper = None
        app.state.sign_retriever = None


def test_unknown_response_marks_no_supported_match():
    result = client.post("/sign/translate", json={"text": "unknown text"}).json()
    assert result["matched"] is False
    assert result["match_type"] == "none"
    assert result["videos"] == []
