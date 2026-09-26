"""Run a controlled tiny-training-set memorization diagnostic."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cnn_lstm_model import CNNBiLSTMClassifier  # noqa: E402
from scripts.dataset import LandmarkDataset  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_best.pt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "cnn_lstm_config.yaml"
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
FEATURE_DIMENSION = 225
MAX_FRAMES = 128
SEED = 42
CLASS_COUNT = 8
SAMPLES_PER_CLASS = 2
SAMPLE_COUNT = CLASS_COUNT * SAMPLES_PER_CLASS
BATCH_SIZE = 4
MAX_EPOCHS = 150


def set_deterministic_seed(seed: int) -> None:
    """Set seeds and deterministic backend options for repeatable diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_configuration() -> tuple[dict[str, Any], float]:
    """Load model architecture settings from checkpoint and LR from YAML."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Model configuration checkpoint not found: {CHECKPOINT_PATH}")
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Training configuration not found: {CONFIG_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_config"), dict):
        raise ValueError(f"Checkpoint does not contain a valid model_config: {CHECKPOINT_PATH}")
    model_config: dict[str, Any] = checkpoint["model_config"]

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        training_config = yaml.safe_load(config_file)
    if not isinstance(training_config, dict):
        raise ValueError(f"Expected a YAML mapping in {CONFIG_PATH}.")

    learning_rate = float(training_config["learning_rate"])
    if learning_rate <= 0:
        raise ValueError(f"Invalid learning_rate in {CONFIG_PATH}: {learning_rate}")
    if int(model_config.get("max_frames", -1)) != MAX_FRAMES:
        raise ValueError(
            f"Checkpoint max_frames must be {MAX_FRAMES}; "
            f"found {model_config.get('max_frames')}."
        )
    return model_config, learning_rate


def select_training_rows(frame: pd.DataFrame) -> tuple[list[int], list[str]]:
    """Select 8 classes with two rows each, preferring the first 16 CSV rows."""
    if "label_id" not in frame.columns:
        raise ValueError(f"{TRAIN_CSV} is missing the required label_id column.")

    labels = frame["label_id"].astype(str)
    first_rows = frame.index[:SAMPLE_COUNT].tolist()
    first_counts = labels.loc[first_rows].value_counts()
    eligible_first_labels = [
        str(label) for label in labels.loc[first_rows].drop_duplicates()
        if int(first_counts.get(label, 0)) >= SAMPLES_PER_CLASS
    ]

    if len(eligible_first_labels) >= CLASS_COUNT:
        source_rows = first_rows
        selected_labels = eligible_first_labels[:CLASS_COUNT]
        selection_description = "selected from the first 16 CSV rows"
    else:
        source_rows = frame.index.tolist()
        all_counts = labels.value_counts()
        selected_labels = [
            label
            for label in sorted(labels.unique())
            if int(all_counts.get(label, 0)) >= SAMPLES_PER_CLASS
        ][:CLASS_COUNT]
        selection_description = "selected from sorted labels across train.csv"

    if len(selected_labels) < CLASS_COUNT:
        raise ValueError(
            f"Could not select {CLASS_COUNT} distinct training classes with "
            f"{SAMPLES_PER_CLASS} samples each from {TRAIN_CSV}; "
            f"found {len(selected_labels)} eligible classes."
        )

    selected_indices: list[int] = []
    for label in selected_labels:
        label_rows = [
            row_index
            for row_index in source_rows
            if str(frame.at[row_index, "label_id"]) == label
        ]
        selected_indices.extend(label_rows[:SAMPLES_PER_CLASS])

    if len(selected_indices) != SAMPLE_COUNT or len(set(selected_labels)) != CLASS_COUNT:
        raise RuntimeError("Deterministic tiny-dataset selection did not produce 16 samples.")
    print(f"Sample selection: {selection_description}")
    print(f"Selected classes: {', '.join(selected_labels)}")
    return selected_indices, selected_labels


def load_tiny_dataset(
    class_to_index: dict[str, int],
    selected_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Load the selected rows with the repository's deterministic preprocessing."""
    if not TRAIN_CSV.is_file():
        raise FileNotFoundError(f"Training split not found: {TRAIN_CSV}")

    try:
        dataset = LandmarkDataset(
            csv_path=TRAIN_CSV,
            max_frames=MAX_FRAMES,
            class_to_index=class_to_index,
            training=False,
            seed=SEED,
            landmark_noise_std=0.0,
            frame_dropout_prob=0.0,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load training split {TRAIN_CSV}: {exc}") from exc

    features: list[torch.Tensor] = []
    padding_masks: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    class_names: list[str] = []
    for sample_number, row_index in enumerate(selected_indices, start=1):
        row = dataset.frame.iloc[row_index]
        landmark_path = str(row["landmark_path"])
        try:
            sample = dataset[row_index]
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Could not load selected training sample {sample_number} "
                f"(CSV row {row_index + 1}, landmark {landmark_path}): {exc}"
            ) from exc

        feature_tensor = sample["features"]
        padding_mask = sample["padding_mask"]
        if feature_tensor.shape != (MAX_FRAMES, FEATURE_DIMENSION):
            raise ValueError(
                f"Unexpected prepared feature shape for {landmark_path}: "
                f"{tuple(feature_tensor.shape)}."
            )
        features.append(feature_tensor)
        padding_masks.append(padding_mask)
        labels.append(sample["label"])
        class_names.append(str(row["label_id"]))

    return (
        torch.stack(features),
        torch.stack(padding_masks),
        torch.stack(labels),
        class_names,
    )


def build_model(
    model_config: dict[str, Any],
    device: torch.device,
) -> CNNBiLSTMClassifier:
    """Construct the existing architecture with an 8-class diagnostic head."""
    return CNNBiLSTMClassifier(
        input_dim=FEATURE_DIMENSION,
        num_classes=CLASS_COUNT,
        cnn_channels_1=int(model_config["cnn_channels_1"]),
        cnn_channels_2=int(model_config["cnn_channels_2"]),
        lstm_hidden_size=int(model_config["lstm_hidden_size"]),
        lstm_layers=int(model_config["lstm_layers"]),
        cnn_dropout=float(model_config["cnn_dropout"]),
        lstm_dropout=float(model_config["lstm_dropout"]),
        classifier_dropout=float(model_config["classifier_dropout"]),
    ).to(device)


def evaluate_accuracy(
    model: CNNBiLSTMClassifier,
    features: torch.Tensor,
    padding_masks: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    """Measure deterministic accuracy over all 16 samples in evaluation mode."""
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device), padding_masks.to(device))
    return float(logits.argmax(dim=1).eq(labels.to(device)).float().mean().item())


def train_diagnostic(
    model: CNNBiLSTMClassifier,
    features: torch.Tensor,
    padding_masks: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float,
    device: torch.device,
) -> tuple[float, float]:
    """Train on only the selected samples and return first/final epoch losses."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    shuffle_generator = torch.Generator().manual_seed(SEED)
    first_epoch_loss: float | None = None
    final_epoch_loss = 0.0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(SAMPLE_COUNT, generator=shuffle_generator)
        loss_total = 0.0
        for start in range(0, SAMPLE_COUNT, BATCH_SIZE):
            batch_indices = permutation[start : start + BATCH_SIZE]
            batch_features = features[batch_indices].to(device)
            batch_masks = padding_masks[batch_indices].to(device)
            batch_labels = labels[batch_indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features, batch_masks)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(batch_indices)

        final_epoch_loss = loss_total / SAMPLE_COUNT
        if epoch == 1:
            first_epoch_loss = final_epoch_loss
        if epoch % 10 == 0:
            accuracy = evaluate_accuracy(
                model,
                features,
                padding_masks,
                labels,
                device,
            )
            print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | training accuracy={accuracy:.4f}")

    if first_epoch_loss is None:
        raise RuntimeError("Diagnostic training completed without recording epoch 1 loss.")
    return first_epoch_loss, final_epoch_loss


def print_final_predictions(
    model: CNNBiLSTMClassifier,
    features: torch.Tensor,
    padding_masks: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    index_to_class: list[str],
    device: torch.device,
) -> float:
    """Print all final predictions and return final training accuracy."""
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(
            model(features.to(device), padding_masks.to(device)),
            dim=1,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    correct = 0
    print("\nFINAL TRAINING PREDICTIONS")
    for sample_index, (expected_index, row_probabilities) in enumerate(
        zip(labels.tolist(), probabilities.cpu())
    ):
        confidence_tensor, predicted_index_tensor = row_probabilities.max(dim=0)
        predicted_index = int(predicted_index_tensor.item())
        confidence = float(confidence_tensor.item())
        predicted_class = index_to_class[predicted_index]
        expected_class = class_names[sample_index]
        is_correct = predicted_index == expected_index
        correct += int(is_correct)
        correctness = "CORRECT" if is_correct else "INCORRECT"
        print(
            f"{sample_index + 1:2d}. expected={expected_class} | "
            f"predicted={predicted_class} | confidence={confidence:.4f} | {correctness}"
        )
    return correct / len(class_names)


def main() -> int:
    """Run the tiny-dataset overfit diagnostic without saving a checkpoint."""
    set_deterministic_seed(SEED)
    if not TRAIN_CSV.is_file():
        raise FileNotFoundError(f"Training split not found: {TRAIN_CSV}")
    train_frame = pd.read_csv(TRAIN_CSV, keep_default_na=False)
    selected_indices, selected_classes = select_training_rows(train_frame)
    class_to_index = {
        class_id: index for index, class_id in enumerate(selected_classes)
    }
    model_config, learning_rate = load_configuration()
    features, padding_masks, labels, sample_class_names = load_tiny_dataset(
        class_to_index,
        selected_indices,
    )
    if len(features) != SAMPLE_COUNT:
        raise ValueError(f"Expected {SAMPLE_COUNT} selected samples, got {len(features)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config, device)

    print("==================================================")
    print("TINY-DATASET OVERFIT DIAGNOSTIC")
    print("Diagnostic only - this is not a production model.")
    print("==================================================")
    print(f"Training split: {TRAIN_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Selected samples: {SAMPLE_COUNT} ({CLASS_COUNT} classes, 2 per class)")
    print(f"Max frames: {MAX_FRAMES}")
    print(f"Feature dimension: {FEATURE_DIMENSION}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Maximum epochs: {MAX_EPOCHS}")
    print(f"Learning rate: {learning_rate}")
    print(f"Seed: {SEED}")
    print(f"Device: {device}")
    print("Training augmentation: disabled (noise=0, frame_dropout=0)")
    print("No checkpoint will be written by this diagnostic.")

    first_epoch_loss, final_epoch_loss = train_diagnostic(
        model,
        features,
        padding_masks,
        labels,
        learning_rate,
        device,
    )
    final_accuracy = print_final_predictions(
        model,
        features,
        padding_masks,
        labels,
        sample_class_names,
        selected_classes,
        device,
    )

    print("\nFINAL SUMMARY")
    print(f"Final training accuracy: {final_accuracy:.4f}")
    print(f"Loss at epoch 1: {first_epoch_loss:.6f}")
    print(f"Loss at final epoch ({MAX_EPOCHS}): {final_epoch_loss:.6f}")
    if final_accuracy >= 0.90:
        print(
            "Interpretation: the model/pipeline can memorize this tiny dataset; "
            "the main problem is likely generalization/training-data difficulty."
        )
    elif final_accuracy < 0.70:
        print(
            "Interpretation: training accuracy remains below 70%; investigate "
            "preprocessing, labels, masking, feature representation, or model "
            "implementation before another full training run."
        )
    else:
        print(
            "Interpretation: accuracy is between the specified thresholds; "
            "this diagnostic does not give a decisive indication."
        )
    print("Diagnostic results are not a production model or a definitive diagnosis.")
    if device.type == "cuda":
        model.to("cpu")
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
