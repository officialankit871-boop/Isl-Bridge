# M8.2 sign video index

## Artifact

`data/manifests/sign_video_index.csv` is a row-preserving index generated from the primary source, `data/manifests/manifest.csv`. It contains these columns in order:

`video_path`, `relative_video_path`, `file_name`, `sentence`, `sentence_normalized`, `class_name`, `label_id`, `gloss`.

`sentence_normalized` lowercases the sentence, trims its edges, and collapses repeated whitespace. It preserves all words and their order. The other source path and mapping fields are carried through unchanged. The gloss CSV is used only to validate exact normalized sentence keys and their gloss values; `label_map.json` validates label IDs and class names.

The index keeps one row per manifest video. Multiple videos for one sentence remain separate rows. The shared gloss is not used as a sentence key:

| Gloss | Sentence |
| --- | --- |
| WHAT YOU DO | what are you doing |
| WHAT YOU DO | what do you do |

The current source data has 687 video rows, 101 sentences, and 100 glosses. It has 99 sentences with multiple video variants and 2 with one variant.

## Missing-path discrepancy

The earlier M8.1 report of 7 missing video paths could not be reproduced from the current manifest. All 687 `video_path` values resolve when joined to the project root (`D:\Isl-Bridge`), and all 687 `relative_video_path` values resolve under `data/raw/archive/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus`. Both checks returned zero missing files. The seven therefore reflect an audit path-resolution or run-context discrepancy, not seven absent videos in the current checkout.

The prior report did not include the seven path values, and no misses occur with either expected root, so those specific rows cannot be identified from the report alone. The builder repeats the project-root existence check and fails with the unresolved path list if a future checkout does contain missing videos. No dataset path is rewritten to compensate.

The earlier M8.1 note naming the shared gloss as `WHAT DO YOU` was incorrect; the manifest and gloss CSV both show `WHAT YOU DO` for the two sentences above.

## Build and validate

Run from the project root:

```powershell
.\.venv310\Scripts\python.exe scripts\build_sign_index.py
.\.venv310\Scripts\python.exe -m pytest -q tests\test_sign_index.py
.\.venv310\Scripts\python.exe -m py_compile scripts\build_sign_index.py
git diff --check
```

The builder validates row count, unique video paths, required values, exact sentence-gloss agreement, label ID/class name agreement, expected shared-gloss collision, preserved variants, and video file existence. The tests use temporary fixture files only and do not require a GPU or alter the dataset.
