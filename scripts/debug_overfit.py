"""Run a small deterministic overfit test for the landmark Transformer."""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import LandmarkDataset, class_mapping, resolve_path  # noqa: E402
from model import TemporalTransformerClassifier  # noqa: E402

TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "debug_overfit_model.pt"

MAX_FRAMES = 128
INPUT_DIM = 225
NUM_CLASSES = 101
NUM_EPOCHS = 100
BATCH_SIZE = 4
SEED = 42


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_landmark(path: Path) -> np.ndarray:
    """Load and validate one landmark array before it enters the dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Landmark file does not exist: {path}")
    try:
        values = np.load(path, allow_pickle=False)
        values = values.astype(np.float32, copy=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not convert landmark file {path}: {exc}") from exc
    if values.ndim != 2 or values.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected (T, {INPUT_DIM}) in {path}, got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError(f"Landmark sequence is empty: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Landmark sequence contains non-finite values: {path}")
    return values


def select_valid_samples(
    frame: pd.DataFrame, mapping: dict[str, int]
) -> tuple[list[int], list[np.ndarray]]:
    """Select the first 20 valid rows while requiring multiple classes."""
    selected_indices: list[int] = []
    selected_arrays: list[np.ndarray] = []
    for index, row in frame.iterrows():
        path = resolve_path(str(row["landmark_path"]))
        try:
            values = validate_landmark(path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping invalid sample {index}: {exc}")
            continue
        selected_indices.append(int(index))
        selected_arrays.append(values)
        if len(selected_indices) == 20:
            break

    if len(selected_indices) < 20:
        raise RuntimeError(
            f"Found only {len(selected_indices)} valid samples; need 20."
        )

    selected_labels = [str(frame.iloc[index]["label_id"]) for index in selected_indices]
    if len(set(selected_labels)) < 2:
        for index, row in frame.iterrows():
            label_id = str(row["label_id"])
            if label_id in selected_labels:
                continue
            path = resolve_path(str(row["landmark_path"]))
            try:
                values = validate_landmark(path)
            except (FileNotFoundError, ValueError):
                continue
            selected_indices[-1] = int(index)
            selected_arrays[-1] = values
            break

    selected_labels = [str(frame.iloc[index]["label_id"]) for index in selected_indices]
    if len(set(selected_labels)) < 2:
        raise RuntimeError("Could not find 20 valid samples from multiple classes.")
    if any(label not in mapping for label in selected_labels):
        raise ValueError("A selected label is missing from the class mapping.")
    return selected_indices, selected_arrays


def main() -> int:
    """Validate data, train on 20 samples, and report the diagnostic result."""
    seed_everything(SEED)
    frame = pd.read_csv(TRAIN_CSV, keep_default_na=False)
    mapping = class_mapping(frame)
    if len(mapping) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(mapping)}.")
    if sorted(mapping.values()) != list(range(NUM_CLASSES)):
        raise ValueError("Class mapping is not an integer range from 0 to 100.")

    selected_indices, selected_arrays = select_valid_samples(frame, mapping)
    selected_labels = [str(frame.iloc[index]["label_id"]) for index in selected_indices]
    mapped_labels = [mapping[label] for label in selected_labels]
    if any(label < 0 or label >= NUM_CLASSES for label in mapped_labels):
        raise ValueError("Selected labels must map to integers from 0 to 100.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of samples: {len(selected_indices)}")
    print(f"Number of unique classes: {len(set(selected_labels))}")
    print(f"Class distribution: {dict(Counter(selected_labels))}")
    print(f"Feature shape: {selected_arrays[0].shape}")
    print(
        f"First sample: path={frame.iloc[selected_indices[0]]['landmark_path']}, "
        f"shape={selected_arrays[0].shape}, min={selected_arrays[0].min():.6g}, "
        f"max={selected_arrays[0].max():.6g}, mean={selected_arrays[0].mean():.6g}, "
        f"std={selected_arrays[0].std():.6g}, "
        f"label_id={selected_labels[0]}, mapped_class_index={mapped_labels[0]}"
    )

    dataset = LandmarkDataset(
        TRAIN_CSV,
        max_frames=MAX_FRAMES,
        class_to_index=mapping,
        training=False,
        seed=SEED,
        landmark_noise_std=0.0,
        frame_dropout_prob=0.0,
    )
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = TemporalTransformerClassifier(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        max_frames=MAX_FRAMES,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)
    print(
        f"Number of model parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0005, weight_decay=0.00001
    )
    labels_for_metrics = list(range(NUM_CLASSES))

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        losses: list[float] = []
        predictions: list[int] = []
        targets: list[int] = []
        for batch in loader:
            features = batch["features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, padding_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.cpu().tolist())

        accuracy = accuracy_score(targets, predictions)
        macro_f1 = f1_score(
            targets,
            predictions,
            labels=labels_for_metrics,
            average="macro",
            zero_division=0,
        )
        print(
            f"Epoch {epoch:03d} | Loss: {np.mean(losses):.6f} | "
            f"Training Accuracy: {accuracy:.4f} | Training Macro F1: {macro_f1:.4f}"
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_index": mapping,
            "model_config": {
                "input_dim": INPUT_DIM,
                "num_classes": NUM_CLASSES,
                "max_frames": MAX_FRAMES,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
            },
            "selected_indices": selected_indices,
        },
        CHECKPOINT_PATH,
    )

    if accuracy >= 0.90:
        print("OVERFIT TEST PASSED: model can memorize the small dataset.")
        print("The basic data/model/loss pipeline is working.")
    else:
        print("OVERFIT TEST FAILED: model cannot memorize the small dataset.")
        print("Investigate labels, landmark values, masking, model forward pass, or loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
