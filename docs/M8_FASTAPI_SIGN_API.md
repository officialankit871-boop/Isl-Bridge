# M8.4 FastAPI sign retrieval API

M8.4 provides exact dataset-grounded sentence retrieval through FastAPI.

## Endpoint and request

`POST /sign/translate` accepts a JSON object containing `text`:

```json
{"text": "Are you free today?"}
```

Empty and whitespace-only text receives FastAPI's standard HTTP 422 validation response. The endpoint appears in the interactive API documentation at `/docs`.

## Successful response

A known sentence returns HTTP 200 with `matched`, the canonical sentence and normalized sentence, `gloss`, `class_name`, `label_id`, `video_count`, and a `videos` array. Every distinct video indexed for that sentence is included.

## Unknown sentence

A syntactically valid sentence absent from the dataset returns HTTP 200 with `matched: false`, its input text and normalized form, null gloss/class/label metadata, `video_count: 0`, and an empty `videos` array.

## Normalization and variants

The API delegates normalization to `SignRetriever`: lowercase, trim and collapse whitespace, and replace `. , ! ? ; :` with spaces. Matching is exact against normalized sentence text. All distinct video variants are returned; the API does not select or stream a video.

Gloss is metadata, never a lookup key. The sentences “what are you doing” and “what do you do” share `WHAT YOU DO` but resolve to separate sentence records and separate video paths.

## Example

```bash
curl -X POST http://127.0.0.1:8000/sign/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Are you free today?"}'
```

## Limitations

This endpoint performs exact dataset lookup only. It does not provide semantic translation, fuzzy search, speech recognition, generated sign language, or arbitrary sentence-to-gloss conversion. Request bodies accept text only; video paths come from the trusted index, and no file-serving endpoint is provided.
