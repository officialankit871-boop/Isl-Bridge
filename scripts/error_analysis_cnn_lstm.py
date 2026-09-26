"""Evaluate and compare the three existing CNN-BiLSTM checkpoints."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cnn_lstm_model import CNNBiLSTMClassifier  # noqa: E402

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
CLASS_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
CHECKPOINTS = {
    "original": {
        "path": PROJECT_ROOT / "checkpoints" / "cnn_lstm_best.pt",
        "feature_dir": PROJECT_ROOT / "data" / "processed" / "landmarks",
        "expected_dim": 225,
    },
    "presence_aware": {
        "path": PROJECT_ROOT / "checkpoints" / "cnn_lstm_presence_best.pt",
        "feature_dir": PROJECT_ROOT / "data" / "processed" / "features_presence_aware",
        "expected_dim": 228,
    },
    "presence_gated": {
        "path": PROJECT_ROOT / "checkpoints" / "cnn_lstm_presence_gated_best.pt",
        "feature_dir": PROJECT_ROOT / "data" / "processed" / "features_presence_gated",
        "expected_dim": 228,
    },
}
REPORT_DIR = PROJECT_ROOT / "reports"
JSON_PATH = REPORT_DIR / "model_error_analysis.json"
CLASS_METRICS_PATH = REPORT_DIR / "per_class_metrics.csv"
CONFUSION_PAIRS_PATH = REPORT_DIR / "confusion_pairs.csv"
COMPARISON_PATH = REPORT_DIR / "model_prediction_comparison.csv"
SPLITS = ("val", "test")
SEED = 42
FEATURE_MANIFEST_COLUMNS = {"video_path", "feature_path", "split"}
SPLIT_COLUMNS = {"landmark_path", "video_path", "label_id", "class_name", "sentence"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def load_split_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = SPLIT_DIR / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Split CSV not found: {path}")
        frame = pd.read_csv(path, keep_default_na=False)
        missing = SPLIT_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["video_path"].map(normalize_path).duplicated().any():
            raise ValueError(f"{path} contains duplicate video_path entries.")
        frames[split] = frame.reset_index(drop=True)
    return frames


def feature_paths_for_model(
    model_name: str,
    feature_dir: Path,
    split_frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Resolve rows by the stable landmark filename key, never by video I/O."""
    if model_name != "original":
        manifest_path = feature_dir / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Feature manifest not found: {manifest_path}")
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        missing = FEATURE_MANIFEST_COLUMNS - set(manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        if manifest["video_path"].map(normalize_path).duplicated().any():
            raise ValueError(f"{manifest_path} contains duplicate video_path values.")
        unknown_splits = sorted(set(manifest["split"].astype(str)) - set(SPLITS) - {"train"})
        if unknown_splits:
            raise ValueError(
                f"{manifest_path} contains unknown split values: {unknown_splits}"
            )
        manifest_by_video = {
            normalize_path(row["video_path"]): row
            for _, row in manifest.iterrows()
        }
    else:
        manifest_by_video = {}

    output: dict[str, pd.DataFrame] = {}
    for split, source in split_frames.items():
        rows: list[dict[str, Any]] = []
        missing_videos: list[str] = []
        for _, row in source.iterrows():
            video_key = normalize_path(row["video_path"])
            if model_name == "original":
                key = Path(
                    str(row["landmark_path"]).strip().replace("\\", "/")
                ).stem
                path = feature_dir / f"{key}.npy"
            else:
                entry = manifest_by_video.get(video_key)
                if entry is None:
                    missing_videos.append(str(row["video_path"]))
                    continue
                if str(entry["split"]) != split:
                    raise ValueError(
                        f"{row['video_path']} has split {entry['split']!r} "
                        f"in {feature_dir.name}, expected {split!r}."
                    )
                relative = Path(str(entry["feature_path"]).replace("\\", "/"))
                path = relative if relative.is_absolute() else PROJECT_ROOT / relative
                path = path.resolve()
                if path.parent != feature_dir.resolve():
                    raise ValueError(
                        f"{model_name} manifest points outside its feature directory: {path}"
                    )
            rows.append(
                {
                    "video_path": str(row["video_path"]),
                    "normalized_video_path": video_key,
                    "feature_path": str(path),
                    "label_id": str(row["label_id"]),
                    "class_name": str(row["class_name"]),
                    "sentence": str(row["sentence"]),
                }
            )
        if missing_videos:
            raise ValueError(
                f"{len(missing_videos)} {split} rows are absent from "
                f"{model_name} feature manifest; examples: {missing_videos[:5]}"
            )
        output[split] = pd.DataFrame(rows)
    if model_name != "original":
        manifest_videos = {
            normalize_path(row["video_path"])
            for _, row in manifest.iterrows()
            if str(row["split"]) in SPLITS
        }
        used_videos = {
            row["normalized_video_path"]
            for frame in output.values()
            for row in frame.to_dict("records")
        }
        if manifest_videos != used_videos:
            raise ValueError(
                f"{model_name} feature manifest does not exactly match val/test."
            )
    return output


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Could not load checkpoint {path}: {exc}") from exc
    required = {"model_state_dict", "model_config", "class_to_index"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"{path} is missing checkpoint fields: {sorted(missing)}")
    mapping = checkpoint["class_to_index"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{path} contains an invalid class_to_index mapping.")
    indices = list(mapping.values())
    if not all(isinstance(index, (int, np.integer)) for index in indices):
        raise ValueError(f"{path} class_to_index values must be integer indices.")
    if sorted(int(index) for index in indices) != list(range(len(mapping))):
        raise ValueError(f"{path} class_to_index values are not contiguous.")
    return checkpoint


def construct_model(
    checkpoint: dict[str, Any],
    expected_dim: int,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CNNBiLSTMClassifier, dict[str, int], int]:
    config = checkpoint["model_config"]
    state = checkpoint["model_state_dict"]
    mapping = {
        str(label): int(index)
        for label, index in checkpoint["class_to_index"].items()
    }
    state_input_dim = int(state["input_norm.weight"].shape[0])
    state_class_count = int(state["classifier.weight"].shape[0])
    if state_input_dim != expected_dim:
        raise ValueError(
            f"{checkpoint_path} has input dimension {state_input_dim}; "
            f"expected {expected_dim}."
        )
    if "input_dim" in config and int(config["input_dim"]) != expected_dim:
        raise ValueError(
            f"{checkpoint_path} config input_dim={config['input_dim']}; "
            f"expected {expected_dim}."
        )
    if state_class_count != len(mapping):
        raise ValueError(
            f"{checkpoint_path} classifier has {state_class_count} outputs "
            f"but mapping has {len(mapping)} labels."
        )
    model = CNNBiLSTMClassifier(
        input_dim=state_input_dim,
        num_classes=state_class_count,
        cnn_channels_1=int(config["cnn_channels_1"]),
        cnn_channels_2=int(config["cnn_channels_2"]),
        lstm_hidden_size=int(config["lstm_hidden_size"]),
        lstm_layers=int(config["lstm_layers"]),
        cnn_dropout=float(config["cnn_dropout"]),
        lstm_dropout=float(config["lstm_dropout"]),
        classifier_dropout=float(config["classifier_dropout"]),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, mapping, int(config["max_frames"])


class EvaluationDataset(Dataset[dict[str, Any]]):
    """Deterministic first-frame crop/zero-pad preparation for evaluation."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_index: dict[str, int],
        max_frames: int,
        input_dim: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.max_frames = max_frames
        self.input_dim = input_dim
        missing = set(self.frame["label_id"].astype(str)) - set(class_to_index)
        if missing:
            raise ValueError(
                f"Evaluation split has labels not in checkpoint mapping: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = Path(str(row["feature_path"]))
        try:
            values = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise RuntimeError(f"Could not load feature file {path}: {exc}") from exc
        if (
            values.ndim != 2
            or values.shape[0] <= 0
            or values.shape[1] != self.input_dim
        ):
            raise ValueError(
                f"Expected non-empty (T, {self.input_dim}) array in {path}; "
                f"found {values.shape}."
            )
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(
                f"Expected finite float32 features in {path}; found {values.dtype}."
            )
        valid_length = min(values.shape[0], self.max_frames)
        sequence = np.zeros((self.max_frames, self.input_dim), dtype=np.float32)
        sequence[:valid_length] = values[:valid_length]
        padding_mask = np.ones(self.max_frames, dtype=np.bool_)
        padding_mask[:valid_length] = False
        return {
            "features": torch.from_numpy(sequence),
            "padding_mask": torch.from_numpy(padding_mask),
            "label": torch.tensor(
                self.class_to_index[str(row["label_id"])], dtype=torch.long
            ),
            "row_index": index,
        }


def predict_split(
    model: nn.Module,
    frame: pd.DataFrame,
    mapping: dict[str, int],
    max_frames: int,
    input_dim: int,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    dataset = EvaluationDataset(frame, mapping, max_frames, input_dim)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    inverse_mapping = {index: label for label, index in mapping.items()}
    output: list[dict[str, Any] | None] = [None] * len(dataset)
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["padding_mask"].to(device)
            labels = batch["label"].to(device)
            autocast_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(features, mask)
                probabilities = torch.softmax(logits.float(), dim=1)
            confidence, predicted_index = probabilities.max(dim=1)
            top_k = probabilities.topk(min(5, probabilities.shape[1]), dim=1).indices
            row_indices = batch["row_index"].tolist()
            for batch_index, row_index in enumerate(row_indices):
                source = frame.iloc[int(row_index)]
                prediction_idx = int(predicted_index[batch_index].item())
                prediction_label = inverse_mapping[prediction_idx]
                top_indices = [
                    int(value)
                    for value in top_k[batch_index].detach().cpu().tolist()
                ]
                output[int(row_index)] = {
                    "video_path": str(source["video_path"]),
                    "true_label": str(source["label_id"]),
                    "predicted_label": prediction_label,
                    "confidence": float(confidence[batch_index].item()),
                    "correct": bool(
                        prediction_label == str(source["label_id"])
                    ),
                    "top_indices": top_indices,
                    "target_index": int(labels[batch_index].item()),
                    "predicted_index": prediction_idx,
                    "top_indices_hit": int(
                        int(labels[batch_index].item()) in top_indices
                    ),
                    "top3_hit": int(
                        int(labels[batch_index].item()) in top_indices[:3]
                    ),
                }
    if any(record is None for record in output):
        raise RuntimeError("Inference did not produce a result for every sample.")
    return [record for record in output if record is not None]


def class_metadata(
    split_frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    if not CLASS_MANIFEST.is_file():
        raise FileNotFoundError(f"Class manifest not found: {CLASS_MANIFEST}")
    manifest = pd.read_csv(CLASS_MANIFEST, keep_default_na=False)
    required = {"label_id", "class_name", "sentence"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"{CLASS_MANIFEST} is missing columns: {sorted(missing)}"
        )
    for _, row in manifest.iterrows():
        label = str(row["label_id"])
        entry = {
            "class_name": str(row["class_name"]),
            "sentence": str(row["sentence"]),
        }
        previous = metadata.get(label)
        if previous is not None and previous != entry:
            raise ValueError(
                f"Inconsistent class/sentence mapping in {CLASS_MANIFEST} "
                f"for {label}: {previous} vs {entry}."
            )
        metadata[label] = entry
    for frame in split_frames.values():
        for _, row in frame.iterrows():
            label = str(row["label_id"])
            entry = {
                "class_name": str(row["class_name"]),
                "sentence": str(row["sentence"]),
            }
            previous = metadata.get(label)
            if previous is not None and previous != entry:
                raise ValueError(
                    f"Inconsistent class/sentence mapping for {label}: "
                    f"{previous} vs {entry}."
                )
            metadata[label] = entry
    if len(metadata) != 101:
        raise ValueError(
            f"Expected sentence metadata for 101 classes, found {len(metadata)}."
        )
    return metadata


def summarize_predictions(
    predictions: list[dict[str, Any]],
    mapping: dict[str, int],
    metadata: dict[str, dict[str, str]],
) -> dict[str, Any]:
    labels = [str(label) for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = np.asarray(
        [label_to_index[record["true_label"]] for record in predictions],
        dtype=np.int64,
    )
    y_pred = np.asarray(
        [label_to_index[record["predicted_label"]] for record in predictions],
        dtype=np.int64,
    )
    y_score = np.asarray([record["confidence"] for record in predictions])
    encoded = y_true * len(labels) + y_pred
    matrix = np.bincount(
        encoded, minlength=len(labels) * len(labels)
    ).reshape(len(labels), len(labels))
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted_counts,
        out=np.zeros(len(labels), dtype=np.float64),
        where=predicted_counts != 0,
    )
    recall = np.divide(
        true_positive,
        true_counts,
        out=np.zeros(len(labels), dtype=np.float64),
        where=true_counts != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(labels), dtype=np.float64),
        where=(precision + recall) != 0,
    )
    class_records: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        support = int(true_counts[index])
        correct = int(matrix[index, index])
        incorrect = support - correct
        class_records.append(
            {
                "class_id": label,
                "class_name": metadata.get(label, {}).get("class_name", ""),
                "sentence": metadata.get(label, {}).get("sentence", ""),
                "support": support,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": correct / support if support else 0.0,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
        )

    confusion_pairs: list[dict[str, Any]] = []
    for true_index, predicted_index in zip(*np.nonzero(matrix)):
        if true_index == predicted_index:
            continue
        true_label = labels[int(true_index)]
        predicted_label = labels[int(predicted_index)]
        confusion_pairs.append(
            {
                "true_class": true_label,
                "true_class_name": metadata.get(true_label, {}).get("class_name", ""),
                "true_sentence": metadata.get(true_label, {}).get("sentence", ""),
                "predicted_class": predicted_label,
                "predicted_class_name": metadata.get(predicted_label, {}).get("class_name", ""),
                "predicted_sentence": metadata.get(predicted_label, {}).get("sentence", ""),
                "count": int(matrix[true_index, predicted_index]),
            }
        )
    confusion_pairs.sort(
        key=lambda record: (
            -record["count"],
            record["true_class"],
            record["predicted_class"],
        )
    )

    prediction_counts = predicted_counts
    prediction_frequency = [
        {
            "predicted_class": labels[index],
            "class_name": metadata.get(labels[index], {}).get("class_name", ""),
            "sentence": metadata.get(labels[index], {}).get("sentence", ""),
            "count": int(prediction_counts[index]),
        }
        for index in range(len(labels))
    ]
    prediction_frequency.sort(
        key=lambda record: (-record["count"], record["predicted_class"])
    )

    total = len(predictions)
    return {
        "summary": {
            "samples": total,
            "accuracy": float(true_positive.sum() / max(1, total)),
            "macro_f1": float(np.mean(f1)),
            "weighted_f1": float(
                np.sum(f1 * true_counts) / max(1, int(true_counts.sum()))
            ),
            "top3_accuracy": float(
                np.mean([record["top3_hit"] for record in predictions])
            ),
            "top5_accuracy": float(
                np.mean([record["top_indices_hit"] for record in predictions])
            ),
            "average_prediction_confidence": float(np.mean(y_score)),
        },
        "per_class_metrics": class_records,
        "confusion_pairs": confusion_pairs,
        "prediction_frequency": prediction_frequency,
        "confusion_matrix": matrix.tolist(),
        "label_order": labels,
    }


def main() -> int:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_frames = load_split_frames()
    metadata = class_metadata(split_frames)
    model_results: dict[str, dict[str, Any]] = {}
    all_class_rows: list[dict[str, Any]] = []
    all_confusion_rows: list[dict[str, Any]] = []
    per_model_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    reference_mapping: dict[str, int] | None = None
    for model_name, model_spec in CHECKPOINTS.items():
        checkpoint_path = model_spec["path"]
        feature_dir = model_spec["feature_dir"]
        expected_dim = int(model_spec["expected_dim"])
        checkpoint = load_checkpoint(checkpoint_path, device)
        model, mapping, max_frames = construct_model(
            checkpoint,
            expected_dim,
            checkpoint_path,
            device,
        )
        if reference_mapping is None:
            reference_mapping = mapping
        elif mapping != reference_mapping:
            raise ValueError(
                f"{model_name} class_to_index mapping differs from the other checkpoints."
            )
        feature_frames = feature_paths_for_model(
            model_name,
            feature_dir,
            split_frames,
        )
        model_results[model_name] = {
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "feature_directory": str(feature_dir.relative_to(PROJECT_ROOT)),
            "input_dimension": expected_dim,
            "max_frames": max_frames,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_best_validation_macro_f1": float(
                checkpoint.get("best_val_macro_f1", 0.0)
            ),
            "splits": {},
        }
        per_model_predictions[model_name] = {}
        for split in SPLITS:
            predictions = predict_split(
                model,
                feature_frames[split],
                mapping,
                max_frames,
                expected_dim,
                device,
                batch_size=16,
            )
            per_model_predictions[model_name][split] = predictions
            result = summarize_predictions(predictions, mapping, metadata)
            model_results[model_name]["splits"][split] = result
            for record in result["per_class_metrics"]:
                all_class_rows.append(
                    {"model": model_name, "split": split, **record}
                )
            for record in result["confusion_pairs"]:
                all_confusion_rows.append(
                    {"model": model_name, "split": split, **record}
                )
            print(
                f"Finished {model_name} {split}: "
                f"accuracy={result['summary']['accuracy']:.4f}, "
                f"macro_f1={result['summary']['macro_f1']:.4f}"
            )
        del model

    assert reference_mapping is not None
    comparison_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        frame = split_frames[split].reset_index(drop=True)
        for row_index, (_, source) in enumerate(frame.iterrows()):
            row: dict[str, Any] = {
                "split": split,
                "video_path": str(source["video_path"]),
                "true_label": str(source["label_id"]),
                "true_class_name": str(source["class_name"]),
                "true_sentence": str(source["sentence"]),
            }
            for model_name in CHECKPOINTS:
                pred = per_model_predictions[model_name][split][row_index]
                predicted_label = pred["predicted_label"]
                row[f"{model_name}_prediction"] = predicted_label
                row[f"{model_name}_prediction_class_name"] = metadata.get(
                    predicted_label, {}
                ).get("class_name", "")
                row[f"{model_name}_prediction_sentence"] = metadata.get(
                    predicted_label, {}
                ).get("sentence", "")
                row[f"{model_name}_confidence"] = pred["confidence"]
            comparison_rows.append(row)

    low_f1: dict[str, dict[str, list[dict[str, Any]]]] = {}
    high_f1: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model_name, result in model_results.items():
        low_f1[model_name] = {}
        high_f1[model_name] = {}
        for split in SPLITS:
            classes = result["splits"][split]["per_class_metrics"]
            low_f1[model_name][split] = sorted(
                classes, key=lambda row: (row["f1"], row["class_id"])
            )[:20]
            high_f1[model_name][split] = sorted(
                classes, key=lambda row: (-row["f1"], row["class_id"])
            )[:20]

    report = {
        "seed": SEED,
        "device": str(device),
        "evaluation": {
            "splits": list(SPLITS),
            "deterministic_temporal_preprocessing": (
                "Keep first max_frames; zero-pad shorter sequences; no augmentation."
            ),
            "average_prediction_confidence": (
                "Mean of each sample's maximum softmax probability."
            ),
        },
        "models": model_results,
        "top_20_lowest_f1_classes": low_f1,
        "top_20_highest_f1_classes": high_f1,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(all_class_rows).to_csv(CLASS_METRICS_PATH, index=False)
    pd.DataFrame(all_confusion_rows).to_csv(CONFUSION_PAIRS_PATH, index=False)
    pd.DataFrame(comparison_rows).to_csv(COMPARISON_PATH, index=False)

    for split in SPLITS:
        print(f"\n{split.upper()} SUMMARY")
        for model_name in CHECKPOINTS:
            summary = model_results[model_name]["splits"][split]["summary"]
            print(
                f"{model_name}: accuracy={summary['accuracy']:.4f} "
                f"macro_f1={summary['macro_f1']:.4f} "
                f"weighted_f1={summary['weighted_f1']:.4f} "
                f"top3={summary['top3_accuracy']:.4f} "
                f"top5={summary['top5_accuracy']:.4f} "
                f"avg_confidence={summary['average_prediction_confidence']:.4f}"
            )
            split_result = model_results[model_name]["splits"][split]
            print(f"  Top 20 confusion pairs ({model_name}):")
            for pair in split_result["confusion_pairs"][:20]:
                print(
                    f"    {pair['true_class']} ({pair['true_class_name']}) -> "
                    f"{pair['predicted_class']} ({pair['predicted_class_name']}): "
                    f"{pair['count']}"
                )
            print(f"  Top 20 most frequently predicted classes ({model_name}):")
            for prediction in split_result["prediction_frequency"][:20]:
                print(
                    f"    {prediction['predicted_class']} "
                    f"({prediction['class_name']}): {prediction['count']}"
                )
            print(f"  Top 20 classes with lowest F1 ({model_name}):")
            for item in low_f1[model_name][split]:
                print(
                    f"    {item['class_id']} ({item['class_name']}): "
                    f"F1={item['f1']:.4f}, support={item['support']}"
                )
            print(f"  Top 20 classes with highest F1 ({model_name}):")
            for item in high_f1[model_name][split]:
                print(
                    f"    {item['class_id']} ({item['class_name']}): "
                    f"F1={item['f1']:.4f}, support={item['support']}"
                )

    print(f"\nJSON report: {JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Per-class metrics: {CLASS_METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Confusion pairs: {CONFUSION_PAIRS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Prediction comparison: {COMPARISON_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
