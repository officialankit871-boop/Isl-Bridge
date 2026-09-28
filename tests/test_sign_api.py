"""Integration tests for exact sentence retrieval through FastAPI."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


# Requests without entering TestClient's lifespan avoid loading the unrelated
# CNN+BiLSTM checkpoint; the sign route lazily reuses its cached retriever.
client = TestClient(app)


def test_sign_translate_known_sentence_returns_every_variant() -> None:
    response = client.post("/sign/translate", json={"text": "Are you free today?"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["matched"] is True
    assert payload["sentence"] == "are you free today"
    assert payload["sentence_normalized"] == "are you free today"
    assert payload["gloss"]
    assert payload["class_name"] == "are_you_free_today"
    assert payload["label_id"]
    assert payload["video_count"] > 0
    assert len(payload["videos"]) == payload["video_count"]
    paths = [video["video_path"] for video in payload["videos"]]
    assert len(paths) == len(set(paths))


def test_sign_translate_unknown_sentence_is_a_normal_miss() -> None:
    response = client.post(
        "/sign/translate",
        json={"text": "This sentence does not exist in the ISL dataset"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["matched"] is False
    assert payload["video_count"] == 0
    assert payload["videos"] == []
    assert payload["gloss"] is None


def test_sign_translate_rejects_empty_and_whitespace_text() -> None:
    for text in ("", " \t\n "):
        response = client.post("/sign/translate", json={"text": text})
        assert response.status_code == 422


def test_sign_translate_uses_retriever_normalization() -> None:
    response = client.post(
        "/sign/translate", json={"text": "  ARE   YOU   FREE   TODAY?  "}
    )
    assert response.status_code == 200
    assert response.json()["matched"] is True
    assert response.json()["sentence_normalized"] == "are you free today"


def test_sign_translate_keeps_shared_gloss_sentences_independent() -> None:
    doing = client.post("/sign/translate", json={"text": "what are you doing"})
    do = client.post("/sign/translate", json={"text": "what do you do"})
    assert doing.status_code == do.status_code == 200
    doing_payload, do_payload = doing.json(), do.json()
    assert doing_payload["sentence"] == "what are you doing"
    assert do_payload["sentence"] == "what do you do"
    assert doing_payload["gloss"] == do_payload["gloss"] == "WHAT YOU DO"
    doing_paths = {video["video_path"] for video in doing_payload["videos"]}
    do_paths = {video["video_path"] for video in do_payload["videos"]}
    assert doing_paths
    assert do_paths
    assert doing_paths.isdisjoint(do_paths)


def test_existing_health_endpoint_still_works() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_documents_sign_endpoint() -> None:
    operation = client.get("/openapi.json").json()["paths"]["/sign/translate"]["post"]
    assert "Retrieve an exact sentence-level Indian Sign Language video mapping, or resolve a reviewed CSV alias to one." in operation["description"]
    assert "The endpoint does not perform fuzzy translation." in operation["description"]
    assert "fuzzy translation" in operation["description"]
