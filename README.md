# ISL Bridge

## Current architecture

Landmarks → CNN+BiLSTM → FastAPI

## Environment

Python 3.10.11

## Setup

Windows commands:

.venv310\Scripts\activate

## Run backend

uvicorn backend.main:app --reload

## Health endpoint

GET /health

## Prediction endpoint

POST /predict

## Swagger

http://127.0.0.1:8000/docs

## Model

cnn_lstm_best.pt

## Input

T × 225 landmark sequence

## Output

prediction + confidence + top-k

## Real-time sign-to-text prototype

Run the local webcam prototype from the project root:

```powershell
.\.venv310\Scripts\python.exe scripts\realtime_inference.py
```

It uses `checkpoints\cnn_lstm_best.pt`, the existing 225-feature extraction
normalization, and manifest-backed sentence labels. The webcam buffer holds
128 frames; inference runs every 5 frames after warm-up. Use `--camera`,
`--confidence-threshold`, `--inference-interval`, `--history-size`, `--cpu`,
or `--show-top5` to adjust runtime behavior. Press `Q`/`ESC` to quit, `R` to
reset the buffer, `S` to toggle top-five output, or `C` to clear the display.
Predictions below the configured confidence threshold are displayed as
“Uncertain”. To enable local, offline Windows speech for newly stabilized
predictions, add `--speak`:

```powershell
.\.venv310\Scripts\python.exe scripts\realtime_inference.py --speak
```

Speech uses the Windows SAPI voice through PowerShell and runs outside the
inference loop. A held sign is spoken once; another stabilized sign or clearing
or resetting recognition allows a later announcement.

## Current limitation

The model is sentence-level sequence classification and is not yet a true continuous sign-language recognizer.

## Future

Webcam → MediaPipe → sequence → API → prediction → TTS

## Text-to-ISL video playback

Run the FastAPI application from the project root:

```powershell
.\.venv310\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000/> to enter an English sentence and browse its
matching indexed ISL videos. The page calls `POST /sign/translate`; video
variants are streamed from the safe indexed route returned by the API. Exact
sentence matches only are supported. See [M8_VIDEO_PLAYBACK](docs/M8_VIDEO_PLAYBACK.md)
for implementation details and tests.
