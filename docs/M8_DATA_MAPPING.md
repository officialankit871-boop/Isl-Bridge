# M8.1 dataset mapping findings

## Inspected sources

The sentence-level dataset is under `data/raw/archive/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/`.

| Source | Schema / observed granularity |
| --- | --- |
| `data/manifests/manifest.csv` | 687 video rows. Columns: `video_path`, `relative_video_path`, `file_name`, `sentence_folder`, `sentence`, `class_name`, `label_id`, `gloss`, `signer`, `file_extension`, `file_size_mb`, `frame_count`, `fps`, `duration_seconds`, `width`, `height`. One row per sentence video variant. |
| `data/manifests/label_map.json` | 101 class-keyed entries with `label_id`, `sentence`, `class_name`, and `gloss`; all 101 agree with the manifest. |
| `corpus_csv_files/ISL Corpus sign glosses.csv` | 101 rows; columns `Sentence`, `SIGN GLOSSES`. This is the direct sentence-level gloss table. |
| `corpus_csv_files/ISL_CSLRT_Corpus details.xlsx` | 792 rows; columns `Sentences`, `File location`. Sentence/file metadata, with multiple locations possible for a sentence. |
| `corpus_csv_files/ISL_CSLRT_Corpus_frame_details.xlsx` | 18,863 rows; columns `Sentence`, `Frames path`. Frame-level path metadata. |
| `corpus_csv_files/ISL_CSLRT_Corpus_word_details.xlsx` | 18,863 rows; columns `Word`, `Frames path`. Word/frame-level metadata, not a sentence-gloss table. |

`manifest.csv` is the safest source for indexing sentence/class/label/gloss/video paths together: its video paths are project-relative and all point to existing files. `label_map.json` is a class lookup that corroborates the same mapping. The gloss CSV is the clearest gloss source and agrees with the manifest. The XLSX tables describe source locations and frame/word assets; they should not be joined to sentences with fuzzy text matching.

## Mapping and counts

- 101 unique sentences, 101 class names, and 101 label IDs.
- 100 unique gloss strings across the 101 sentence entries.
- 687 sentence-level videos; every required sentence, gloss, class name, label ID, and path field is populated.
- All 687 manifest video paths exist on disk.
- Videos per sentence: minimum 1, median 7, mean 6.80, maximum 9.
- 99 sentence classes have multiple videos; 2 have one video.
- The gloss CSV has no duplicate sentence-gloss rows and no sentence assigned conflicting glosses.
- Manifest sentence-gloss pairs repeat across video variants (99 such multi-video sentence groups); these are expected one-to-many video mappings, not duplicate gloss-table entries.
- `WHAT ARE YOU DOING` and `WHAT DO YOU DO` both map to `WHAT DO YOU`. This is a genuine shared gloss, so reverse lookup from gloss alone is ambiguous.
- No manifest-to-gloss CSV mismatch or label-map-to-manifest mismatch was found using exact sentence keys.

Example mappings:

| Sentence | Gloss | Example video |
| --- | --- | --- |
| are you free today | YOU FREE TODAY | `Videos_Sentence_Level/are you free today/are you free today (2).mp4` |
| are you hiding something | YOU HIDE SOMETHING | `Videos_Sentence_Level/are you hiding something/are you hiding something.mp4` |
| bring water for me | BRING WATER ME | `Videos_Sentence_Level/bring water for me/bring water for me.mp4` |

Each example sentence can have more than one video; the displayed path is one variant, not a unique canonical clip.

## Recommendation for M8.2

Build a small searchable index directly from the existing manifest, keeping one record per video and retaining `sentence`, `gloss`, `class_name`, `label_id`, and `video_path`. Use the gloss CSV as an exact validation source and `label_map.json` as a class-level cross-check. Treat gloss lookup as one-to-many, since distinct sentences can share a gloss. Preserve all video variants for a sentence and avoid inventing sentence-to-video choices or fuzzy joins. This is sufficient for a first sentence → gloss → ISL video index; retrieval itself is outside M8.1.

## Lightweight audit command

From the project root, run:

```powershell
.\.venv310\Scripts\python.exe scripts\m8_dataset_analysis.py
```

The report script reads the original inputs only and writes no derived dataset files.
