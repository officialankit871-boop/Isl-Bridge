# M8.5 Text-to-ISL video playback

## Architecture

The lightweight frontend is plain HTML, CSS, and JavaScript served by the existing FastAPI application. The browser posts sentence text to `POST /sign/translate`; the API returns the exact match, gloss, and all video variants. The page changes video sources locally when the user navigates variants, without repeating translation requests.

## Frontend files

- `frontend/index.html` contains the accessible text form, status, result card, and HTML5 video player.
- `frontend/css/style.css` provides the responsive layout and controls.
- `frontend/js/app.js` handles validation, loading/error states, translation, and local variant navigation.

FastAPI serves `/` and the two asset paths from these files. Same-origin hosting means no additional CORS configuration is required.

## Video serving and security

`POST /sign/translate` supplies a stable SHA-256 video identifier and same-origin `video_url` for each indexed row. `GET /sign/video/{video_id}` resolves identifiers only through a cached map built from the already-loaded `SignRetriever`; it never accepts a client-provided filesystem path. The resulting relative path is resolved under the configured dataset root, checked to remain within that root, and served only when it is an existing file. Unknown identifiers and traversal attempts return 404. No arbitrary file or directory serving is enabled.

## Run and test

From the project root:

```powershell
.\.venv310\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/`. Enter a sentence such as “what are you doing”, select **Translate to ISL**, then use the video controls and Previous/Next to review variants. The interactive backend API docs remain at `/docs`.

Run focused tests with:

```powershell
.\.venv310\Scripts\python.exe -m pytest -q tests\test_sign_video_api.py tests\test_frontend_routes.py
```

## Limitations

The application performs exact dataset-grounded sentence retrieval. Unknown sentences receive a no-match message. It does not provide fuzzy matching, semantic translation, speech recognition, generative sign synthesis, or video generation. Browser playback requires the indexed dataset files to be present on the backend host.
