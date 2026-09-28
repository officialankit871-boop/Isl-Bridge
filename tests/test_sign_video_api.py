"""Tests for the safe identifier-based indexed video serving route."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.routes_sign import get_sign_retriever
from backend.main import app


client = TestClient(app)


def test_known_indexed_video_is_streamed() -> None:
    translated = client.post("/sign/translate", json={"text": "what are you doing"})
    assert translated.status_code == 200
    item = translated.json()["videos"][0]
    assert len(item["video_id"]) == 64
    assert item["video_url"] == f"/sign/video/{item['video_id']}"

    response = client.get(item["video_url"])
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert len(response.content) > 0


def test_unknown_video_identifier_returns_404() -> None:
    response = client.get("/sign/video/" + "0" * 64)
    assert response.status_code == 404


def test_path_traversal_cannot_select_unindexed_file() -> None:
    response = client.get("/sign/video/..%2F..%2FREADME.md")
    assert response.status_code == 404
    assert b"ISL Bridge" not in response.content


def test_frontend_is_served_by_fastapi() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "ISL Bridge" in response.text
    assert "Translate to ISL" in response.text


def test_frontend_assets_are_served_by_fastapi() -> None:
    css = client.get("/css/style.css")
    js = client.get("/js/app.js")
    assert css.status_code == 200
    assert "@media" in css.text
    assert js.status_code == 200
    assert "/sign/translate" in js.text


def test_sign_translation_still_returns_browser_video_urls() -> None:
    response = client.post("/sign/translate", json={"text": "what are you doing"})
    assert response.status_code == 200
    videos = response.json()["videos"]
    assert videos
    assert all(video["video_url"].startswith("/sign/video/") for video in videos)


def test_existing_health_endpoint_still_works() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_video_map_reuses_loaded_retriever() -> None:
    assert get_sign_retriever() is get_sign_retriever()
