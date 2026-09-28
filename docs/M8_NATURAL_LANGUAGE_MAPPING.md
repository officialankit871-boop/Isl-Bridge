# M8.6 controlled text mapping

`TextMapper` resolves user text against the existing sentence index. It first applies the same normalization as `SignRetriever` and checks for an exact normalized sentence. Only after an exact miss does it consult `data/manifests/text_aliases.csv`. A successful alias resolves to a canonical sentence in `sign_video_index.csv`, which is then passed to the existing `SignRetriever` for video variants.

## Alias CSV format

The UTF-8 CSV has two columns: `alias,canonical_sentence`. Each normalized alias must be unique, and every canonical sentence must already exist in the sign video index. Invalid or duplicate mappings prevent mapper startup. Add an alias only after its equivalence to the canonical sentence has been verified from an authoritative project source. The current file contains only its header because the inspected project sources do not establish any safe alternate natural-language sentence forms.

Example format for a future reviewed entry (illustrative only; not a verified mapping and not present in the production CSV):

```csv
alias,canonical_sentence
<verified alternate wording>,<exact indexed sentence>
```

The API preserves the existing response fields and adds `match_type` with values `exact`, `alias`, or `none`. Exact and alias result examples:

```json
{"matched": true, "match_type": "exact", "sentence": "are you free today"}
{"matched": true, "match_type": "alias", "sentence": "<canonical indexed sentence>"}
{"matched": false, "match_type": "none", "sentence": "unknown input"}
```

This is a finite curated mapping layer. It does not infer meaning or use fuzzy matching, embeddings, LLMs, translation services, speech recognition, or generated videos.
