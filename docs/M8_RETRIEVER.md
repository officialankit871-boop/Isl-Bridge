# M8.3 sentence retriever

## Purpose

`backend/services/sign_retriever.py` provides lightweight, deterministic lookup from an ISL sentence to its indexed sentence metadata and video files. It is a local service module; M8.3 adds no API route.

## Input index

The retriever reads `data/manifests/sign_video_index.csv` once during initialization. It validates the required columns and keeps the rows in memory. The CSV is read-only from this service. The default path is relative to the project root; tests and other callers may pass an explicit index path.

## Normalization and matching

Queries are lowercased, trimmed, and split/joined to collapse repeated whitespace. The sentence-ending punctuation characters `. , ! ? ; :` are replaced with spaces before whitespace is collapsed, so punctuation cannot accidentally join neighboring words. Words and their order are preserved. Matching is exact against the normalized `sentence_normalized` index key. A normalized sentence with conflicting gloss, class name, label ID, or source sentence causes initialization to fail clearly.

## Multiple videos and shared glosses

Every distinct `video_path` for a sentence is retained and returned in `videos`; `video_count` equals the returned list length. Sentence lookup never uses gloss as its key. Consequently, “what are you doing” and “what do you do” remain independent records even though both have gloss `WHAT YOU DO`.

## Usage

```python
from backend.services.sign_retriever import SignRetriever

retriever = SignRetriever()
retriever.find_sentence("Are you free today?")
retriever.get_video_variants("what are you doing")
retriever.get_by_sentence("what do you do")
retriever.find_sentence("This sentence does not exist in the ISL dataset")
```

Successful results include `matched`, `sentence`, `sentence_normalized`, `gloss`, `class_name`, `label_id`, `videos`, and `video_count`. Unknown queries return `matched: false`, null metadata, and an empty video list without raising an exception.

## Limitations

**M8.3 performs exact sentence retrieval only.** It does not perform fuzzy matching, semantic search, translation, speech recognition, or generative sign synthesis. It does not use embeddings, an LLM, a vector database, or machine learning.
