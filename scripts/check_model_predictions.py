"""Compare deterministic train and validation predictions for the CNN+BiLSTM."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.inference_service import load_manifest_class_mappings  # noqa: E402
from scripts.cnn_lstm_model import CNNBiLSTMClassifier  # noqa: E402
from scripts.dataset import LandmarkDataset  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_best.pt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "cnn_lstm_config.yaml"
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
VALIDATION_CSV = PROJECT_ROOT / "data" / "splits" / "val.csv"
SAMPLE_COUNT = 20
FEATURE_DIMENSION = 225
MAX_SEQUENCE_LENGTH = 4096


def load_checkpoint() -> dict[str, Any]:
    """Load and validate the CNN+BiLSTM checkpoint."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: expected a dictionary at {CHECKPOINT_PATH}.")
    for required_key in ("model_config", "class_to_index", "model_state_dict"):
        if required_key not in checkpoint:
            raise KeyError(f"Checkpoint is missing required key '{required_key}'.")

    if not isinstance(checkpoint["model_config"], dict):
        raise ValueError("Checkpoint model_config must be a dictionary.")
    if not isinstance(checkpoint["class_to_index"], dict):
        raise ValueError("Checkpoint class_to_index must be a dictionary.")
    return checkpoint


def validate_config_file() -> None:
    """Ensure the project's CNN+BiLSTM configuration file is readable."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"CNN+BiLSTM configuration file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {CONFIG_PATH}.")


def reverse_class_mapping(class_to_index: dict[str, int]) -> dict[int, str]:
    """Build a validated model-index-to-label mapping."""
    index_to_class: dict[int, str] = {}
    for class_id, raw_index in class_to_index.items():
        if not isinstance(class_id, str):
            raise ValueError("Checkpoint class labels must be strings.")
        index = int(raw_index)
        if index in index_to_class:
            raise ValueError(f"Duplicate class index {index} in checkpoint mapping.")
        index_to_class[index] = class_id
    return index_to_class


def load_sentence_mappings() -> tuple[dict[str, str], dict[str, str]]:
    """Load sentence mappings from the manifest, continuing without them if invalid."""
    try:
        return load_manifest_class_mappings()
    except (OSError, ValueError) as exc:
        print(f"Warning: sentence mapping unavailable from data/manifests/manifest.csv: {exc}")
        return {}, {}


def sentence_for_label(
    class_id: str,
    class_name_to_sentence: dict[str, str],
    label_id_to_class_name: dict[str, str],
) -> str | None:
    """Return a manifest sentence only when both class mapping steps resolve."""
    class_name = label_id_to_class_name.get(class_id)
    if class_name is None:
        return None
    return class_name_to_sentence.get(class_name)


def build_model(
    config: dict[str, Any],
    class_count: int,
    device: torch.device,
) -> CNNBiLSTMClassifier:
    """Reconstruct the trained architecture from checkpoint configuration."""
    model = CNNBiLSTMClassifier(
        input_dim=FEATURE_DIMENSION,
        num_classes=class_count,
        cnn_channels_1=int(config["cnn_channels_1"]),
        cnn_channels_2=int(config["cnn_channels_2"]),
        lstm_hidden_size=int(config["lstm_hidden_size"]),
        lstm_layers=int(config["lstm_layers"]),
        cnn_dropout=float(config["cnn_dropout"]),
        lstm_dropout=float(config["lstm_dropout"]),
        classifier_dropout=float(config["classifier_dropout"]),
    ).to(device)
    return model


def diagnose_split(
    split_name: str,
    csv_path: Path,
    model: CNNBiLSTMClassifier,
    class_to_index: dict[str, int],
    index_to_class: dict[int, str],
    max_frames: int,
    device: torch.device,
    class_name_to_sentence: dict[str, str],
    label_id_to_class_name: dict[str, str],
) -> dict[str, Any]:
    """Evaluate the first up-to-20 split rows in stable CSV order."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"{split_name} split not found: {csv_path}")

    try:
        dataset = LandmarkDataset(
            csv_path=csv_path,
            max_frames=max_frames,
            class_to_index=class_to_index,
            training=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load {split_name} split {csv_path}: {exc}") from exc

    sample_count = min(SAMPLE_COUNT, len(dataset))
    if sample_count == 0:
        raise ValueError(f"{split_name} split contains no samples: {csv_path}")

    correct_count = 0
    confidence_total = 0.0
    incorrect_predictions: list[tuple[str, str, float]] = []
    model.eval()
    print(f"\n---------------- {split_name} ----------------")

    for dataset_index in range(sample_count):
        row = dataset.frame.iloc[dataset_index]
        landmark_path = str(row["landmark_path"])
        try:
            sample = dataset[dataset_index]
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Could not evaluate {split_name} sample {dataset_index + 1} "
                f"(landmark: {landmark_path}): {exc}"
            ) from exc

        expected_index = int(sample["label"].item())
        expected_class = index_to_class.get(expected_index)
        if expected_class is None:
            raise ValueError(
                f"Expected class index {expected_index} from {landmark_path} "
                "is absent from the checkpoint mapping."
            )

        features = sample["features"].unsqueeze(0).to(device)
        padding_mask = sample["padding_mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(features, padding_mask)
            probabilities = torch.softmax(logits[0], dim=0)
        confidence_tensor, predicted_index_tensor = probabilities.max(dim=0)
        predicted_index = int(predicted_index_tensor.item())
        predicted_class = index_to_class.get(predicted_index)
        if predicted_class is None:
            raise ValueError(
                f"Predicted class index {predicted_index} is absent from the checkpoint mapping."
            )

        confidence = float(confidence_tensor.item())
        is_correct = predicted_class == expected_class
        correct_count += int(is_correct)
        confidence_total += confidence
        sentence = sentence_for_label(
            predicted_class,
            class_name_to_sentence,
            label_id_to_class_name,
        )
        sentence_text = f' | sentence="{sentence}"' if sentence else " | sentence=<unmapped>"
        correctness = "CORRECT" if is_correct else "INCORRECT"
        print(
            f"{dataset_index + 1}. expected={expected_class} | "
            f"predicted={predicted_class} | {correctness} | "
            f"confidence={confidence:.4f}{sentence_text}"
        )
        if not is_correct:
            incorrect_predictions.append((expected_class, predicted_class, confidence))

    incorrect_count = sample_count - correct_count
    accuracy = correct_count / sample_count
    average_confidence = confidence_total / sample_count
    print(f"\n{split_name} SUMMARY")
    print(f"Samples: {sample_count}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {incorrect_count}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average confidence: {average_confidence:.4f}")
    print(f"\n{split_name} INCORRECT PREDICTIONS")
    if incorrect_predictions:
        for expected_class, predicted_class, confidence in incorrect_predictions:
            print(
                f"{expected_class} -> {predicted_class} -> "
                f"confidence={confidence:.4f}"
            )
    else:
        print("None")

    return {
        "samples": sample_count,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "accuracy": accuracy,
        "average_confidence": average_confidence,
    }


def print_diagnosis(train_accuracy: float, validation_accuracy: float) -> None:
    """Print a cautious indication based on the sampled split accuracies."""
    gap = train_accuracy - validation_accuracy
    print("\n---------------- DIAGNOSIS ----------------")
    if train_accuracy < 0.5 and validation_accuracy < 0.5:
        print(
            "Model is underfitting or the feature/model/training setup needs investigation."
        )
    elif gap >= 0.2:
        print("Evidence of overfitting/generalization gap.")
    else:
        print("No large train-validation accuracy gap observed in this sample.")
    print(
        "Diagnostic indication only; these results use at most 20 samples per split "
        "and are not definitive."
    )


def main() -> int:
    """Run the deterministic train-versus-validation prediction diagnostic."""
    validate_config_file()
    checkpoint = load_checkpoint()
    model_config: dict[str, Any] = checkpoint["model_config"]
    class_to_index: dict[str, int] = checkpoint["class_to_index"]
    index_to_class = reverse_class_mapping(class_to_index)
    if len(class_to_index) != len(index_to_class):
        raise ValueError("Checkpoint class mapping is invalid.")

    max_frames = int(model_config["max_frames"])
    if max_frames <= 0:
        raise ValueError(f"Invalid max_frames value in checkpoint: {max_frames}")
    if max_frames > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Checkpoint max_frames={max_frames} exceeds the supported "
            f"sequence length {MAX_SEQUENCE_LENGTH}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config, len(class_to_index), device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(f"Checkpoint architecture mismatch: {exc}") from exc
    model.eval()

    class_name_to_sentence, label_id_to_class_name = load_sentence_mappings()
    print("==================================================")
    print("CNN-BiLSTM MODEL DIAGNOSTIC")
    print("==================================================")
    print(f"Checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Device: {device}")
    print(f"Classes: {len(class_to_index)}")
    print("\nMODEL CONFIG")
    print(yaml.safe_dump(model_config, sort_keys=False).rstrip())
    print(
        "\nSample selection: first 20 rows in each split CSV "
        "(or all rows if fewer); fixed CSV order, no shuffling."
    )

    train_summary = diagnose_split(
        "TRAIN",
        TRAIN_CSV,
        model,
        class_to_index,
        index_to_class,
        max_frames,
        device,
        class_name_to_sentence,
        label_id_to_class_name,
    )
    validation_summary = diagnose_split(
        "VALIDATION",
        VALIDATION_CSV,
        model,
        class_to_index,
        index_to_class,
        max_frames,
        device,
        class_name_to_sentence,
        label_id_to_class_name,
    )
    print_diagnosis(train_summary["accuracy"], validation_summary["accuracy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
