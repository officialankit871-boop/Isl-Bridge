"""Print a lightweight audit of sentence, gloss, label, and video mappings."""

from __future__ import annotations

import csv
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/raw/archive/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus"
MANIFEST = ROOT / "data/manifests/manifest.csv"
LABEL_MAP = ROOT / "data/manifests/label_map.json"
GLOSS_CSV = DATASET / "corpus_csv_files/ISL Corpus sign glosses.csv"
XLSX_FILES = (
    "ISL_CSLRT_Corpus details.xlsx",
    "ISL_CSLRT_Corpus_frame_details.xlsx",
    "ISL_CSLRT_Corpus_word_details.xlsx",
)
NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def xlsx_sheets(path: Path) -> list[tuple[str, list[str], int]]:
    """Read workbook sheet names, header cells, and row counts without dependencies."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", NS):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{NS['main']}}}t")))

        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relationships.findall("pkg:Relationship", NS)
        }
        result = []
        for sheet in workbook.findall("main:sheets/main:sheet", NS):
            target = targets[sheet.attrib[f"{{{NS['rel']}}}id"]]
            sheet_path = PurePosixPath(target.lstrip("/"))
            if not target.startswith("/") and not str(sheet_path).startswith("xl/"):
                sheet_path = PurePosixPath("xl") / sheet_path
            sheet_root = ElementTree.fromstring(archive.read(str(sheet_path)))
            rows = sheet_root.findall("main:sheetData/main:row", NS)
            headers: list[str] = []
            if rows:
                for cell in rows[0].findall("main:c", NS):
                    value = cell.find("main:v", NS)
                    inline = cell.find("main:is", NS)
                    if cell.attrib.get("t") == "s" and value is not None:
                        headers.append(shared[int(value.text or "0")])
                    elif inline is not None:
                        headers.append("".join(n.text or "" for n in inline.iter(f"{{{NS['main']}}}t")))
                    else:
                        headers.append(value.text if value is not None and value.text else "")
            result.append((sheet.attrib["name"], headers, max(0, len(rows) - 1)))
        return result


def clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def main() -> int:
    required = (MANIFEST, LABEL_MAP, GLOSS_CSV)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input file(s) missing: " + ", ".join(missing))

    manifest = read_csv(MANIFEST)
    gloss_rows = read_csv(GLOSS_CSV)
    labels = json.loads(LABEL_MAP.read_text(encoding="utf-8-sig"))
    print("M8 dataset mapping audit (exact text keys; no fuzzy matching)")
    print(f"Manifest columns: {', '.join(manifest[0].keys()) if manifest else '(none)'}")
    print(f"Gloss CSV columns: {', '.join(gloss_rows[0].keys()) if gloss_rows else '(none)'}")

    gloss_column = "SIGN GLOSSES"
    gloss_pairs = Counter((clean(row.get("Sentence")), clean(row.get(gloss_column))) for row in gloss_rows)
    gloss_by_sentence: dict[str, set[str]] = defaultdict(set)
    for sentence, gloss in gloss_pairs:
        if sentence and gloss:
            gloss_by_sentence[sentence].add(gloss)

    by_sentence: dict[str, list[dict[str, str]]] = defaultdict(list)
    manifest_pairs: Counter[tuple[str, str]] = Counter()
    blanks = Counter()
    absent_video_paths: list[str] = []
    for row in manifest:
        sentence = clean(row.get("sentence"))
        gloss = clean(row.get("gloss"))
        video_path = clean(row.get("video_path"))
        by_sentence[sentence].append(row)
        manifest_pairs[(sentence, gloss)] += 1
        for field in ("sentence", "gloss", "class_name", "label_id", "video_path"):
            if not clean(row.get(field)):
                blanks[field] += 1
        if video_path and not (ROOT / Path(video_path)).is_file():
            absent_video_paths.append(video_path)

    counts = [len(rows) for sentence, rows in by_sentence.items() if sentence]
    multiple = [(sentence, rows) for sentence, rows in by_sentence.items() if sentence and len(rows) > 1]
    duplicate_gloss_pairs = [(pair, count) for pair, count in gloss_pairs.items() if count > 1]
    gloss_csv_missing = sum(
        not clean(row.get("Sentence")) or not clean(row.get(gloss_column))
        for row in gloss_rows
    )
    sentences_missing_gloss = sorted(set(by_sentence) - set(gloss_by_sentence))
    conflicting_gloss_sentences = {
        sentence: glosses for sentence, glosses in gloss_by_sentence.items() if len(glosses) > 1
    }
    gloss_mismatches = [
        (sentence, sorted(glosses), sorted(gloss_by_sentence.get(sentence, set())))
        for sentence, glosses in {
            sentence: {clean(row.get("gloss")) for row in rows if clean(row.get("gloss"))}
            for sentence, rows in by_sentence.items()
        }.items()
        if glosses != gloss_by_sentence.get(sentence, set())
    ]
    repeated_glosses: dict[str, set[str]] = defaultdict(set)
    for sentence, gloss in gloss_pairs:
        if sentence and gloss:
            repeated_glosses[gloss].add(sentence)
    glosses_shared_by_sentences = {
        gloss: sentences for gloss, sentences in repeated_glosses.items() if len(sentences) > 1
    }
    manifest_duplicate_pairs = {
        pair: count for pair, count in manifest_pairs.items() if pair[0] and pair[1] and count > 1
    }
    manifest_by_label = {clean(row.get("label_id")): row for row in manifest if clean(row.get("label_id"))}
    label_map_mismatches = [
        label_id
        for label_id, label in labels.items()
        if label_id not in manifest_by_label
        or clean(label.get("sentence")) != clean(manifest_by_label[label_id].get("sentence"))
        or clean(label.get("class_name")) != clean(manifest_by_label[label_id].get("class_name"))
        or clean(label.get("gloss")) != clean(manifest_by_label[label_id].get("gloss"))
    ]

    print(f"Unique sentences: {len(by_sentence)}")
    print(f"Unique glosses: {len({gloss for _, gloss in gloss_pairs if gloss})}")
    print(f"Video rows: {len(manifest)}")
    print(
        "Videos per sentence: "
        f"min={min(counts) if counts else 0}, "
        f"median={statistics.median(counts) if counts else 0}, "
        f"mean={statistics.mean(counts) if counts else 0:.2f}, "
        f"max={max(counts) if counts else 0}"
    )
    print(f"Sentences with multiple videos: {len(multiple)}")
    print(f"Missing required manifest values: {dict(blanks)}")
    print(f"Gloss CSV rows with missing sentence/gloss: {gloss_csv_missing}")
    print(f"Manifest sentences without an exact gloss-table match: {len(sentences_missing_gloss)}")
    print(f"Manifest video paths not found: {len(absent_video_paths)}")
    print(f"Gloss CSV duplicate sentence-gloss rows: {len(duplicate_gloss_pairs)}")
    print(f"Repeated manifest sentence-gloss groups across video variants: {len(manifest_duplicate_pairs)}")
    print(f"Sentences with conflicting glosses: {len(conflicting_gloss_sentences)}")
    print(f"Manifest sentence-gloss pairs not matching gloss CSV: {len(gloss_mismatches)}")
    print(f"Label-map entries: {len(labels)}; unique label IDs: {len({v.get('label_id') for v in labels.values()})}")
    print(f"Label-map entries inconsistent with manifest: {len(label_map_mismatches)}")
    print(f"Gloss shared by multiple sentences: {len(glosses_shared_by_sentences)}")

    for filename in XLSX_FILES:
        path = DATASET / "corpus_csv_files" / filename
        if not path.is_file():
            print(f"XLSX missing: {path.relative_to(ROOT)}")
            continue
        try:
            for sheet, columns, rows in xlsx_sheets(path):
                print(f"XLSX {filename} / {sheet}: {rows} data rows; columns={columns}")
        except (OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            print(f"XLSX could not be read: {filename}: {exc}")

    print("Example sentence -> gloss -> video mappings:")
    for sentence, rows in list(by_sentence.items())[:3]:
        if rows:
            row = rows[0]
            print(f"  {sentence} -> {clean(row.get('gloss'))} -> {clean(row.get('video_path'))}")
    if glosses_shared_by_sentences:
        print("Glosses shared across distinct sentences:")
        for gloss, sentences in glosses_shared_by_sentences.items():
            print(f"  {gloss}: {' | '.join(sorted(sentences))}")

    sufficient = (
        bool(manifest)
        and not blanks
        and not absent_video_paths
        and not gloss_mismatches
        and not label_map_mismatches
    )
    print(
        "Existing manifest sufficient for a searchable M8 sentence-gloss-video index: "
        f"{'yes' if sufficient else 'no'}"
    )
    print("Note: repeated manifest sentence-gloss pairs correspond to video variants, not gloss-table duplicates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
