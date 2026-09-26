"""Evaluate the current CNN+BiLSTM checkpoint on the validation/test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cnn_lstm_model import CNNBiLSTMClassifier  # noqa: E402
from scripts.dataset import LandmarkDataset  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_best.pt"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "cnn_lstm_evaluation.json"


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Calculate top-k accuracy for a batch of logits."""
    predictions = logits.topk(k=k, dim=1).indices
    matches = predictions.eq(labels.unsqueeze(1)).any(dim=1)
    return float(matches.float().mean().item())


def evaluate() -> dict[str, Any]:
    """Load the current checkpoint and evaluate it on the test split."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    class_to_index = checkpoint["class_to_index"]
    num_classes = len(class_to_index)

    if num_classes != 101:
        raise ValueError(f"Expected 101 classes, found {num_classes}.")

    test_csv = PROJECT_ROOT / "data" / "splits" / "test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"Test split not found: {test_csv}")

    dataset = LandmarkDataset(
        test_csv,
        config["max_frames"],
        class_to_index,
        training=False,
        seed=config["seed"],
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNBiLSTMClassifier(
        input_dim=225,
        num_classes=num_classes,
        cnn_channels_1=config["cnn_channels_1"],
        cnn_channels_2=config["cnn_channels_2"],
        lstm_hidden_size=config["lstm_hidden_size"],
        lstm_layers=config["lstm_layers"],
        cnn_dropout=config["cnn_dropout"],
        lstm_dropout=config["lstm_dropout"],
        classifier_dropout=config["classifier_dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(features, padding_mask)
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    predictions = logits.argmax(dim=1).numpy()
    targets = labels.numpy()

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "top_3_accuracy": top_k_accuracy(logits, labels, 3),
        "top_5_accuracy": top_k_accuracy(logits, labels, 5),
        "classification_report": classification_report(
            targets,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    """Run the evaluation and print summary metrics."""
    metrics = evaluate()
    print("CNN + BiLSTM evaluation Summary")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
