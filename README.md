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
“Uncertain”; no speech output is produced.

## Current limitation

The model is sentence-level sequence classification and is not yet a true continuous sign-language recognizer.

## Future

Webcam → MediaPipe → sequence → API → prediction → TTS
