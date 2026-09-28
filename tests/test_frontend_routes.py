"""Route and user-flow checks for the served Text-to-ISL frontend."""

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_home_page_and_assets_load() -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="sentence-input"' in page.text
    assert client.get("/css/style.css").status_code == 200
    assert client.get("/js/app.js").status_code == 200


def test_match_and_unknown_flows_return_expected_api_data() -> None:
    matched = client.post("/sign/translate", json={"text": "what are you doing"}).json()
    unknown = client.post("/sign/translate", json={"text": "this sentence does not exist"}).json()
    assert matched["matched"] is True
    assert matched["gloss"] == "WHAT YOU DO"
    assert matched["videos"][0]["video_url"].startswith("/sign/video/")
    assert unknown["matched"] is False
    assert unknown["videos"] == []
