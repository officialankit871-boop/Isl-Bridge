"""Evaluate a trained ISL Transformer checkpoint on the test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import LandmarkDataset  # noqa: E402
from model import TemporalTransformerClassifier  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
CONFUSION_PATH = PROJECT_ROOT / "plots" / "confusion_matrix.png"


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Calculate top-k accuracy."""
    predictions = logits.topk(k, dim=1).indices
    return float(predictions.eq(labels.unsqueeze(1)).any(dim=1).float().mean())


def main() -> int:
    """Evaluate the best checkpoint using test data only."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint: dict[str, Any] = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    mapping: dict[str, int] = checkpoint["class_to_index"]
    test_csv = PROJECT_ROOT / "data" / "splits" / "test.csv"
    test_frame = pd.read_csv(test_csv)
    dataset = LandmarkDataset(test_csv, config["max_frames"], mapping, training=False, seed=config["seed"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformerClassifier(
        input_dim=225, num_classes=len(mapping), max_frames=config["max_frames"],
        d_model=config["d_model"], nhead=config["nhead"], num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"], dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            logits_list.append(model(
                batch["features"].to(device), batch["padding_mask"].to(device)
            ).cpu())
            labels_list.append(batch["label"])
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    predictions = logits.argmax(dim=1).numpy()
    targets = labels.numpy()
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(targets, predictions, average="weighted", zero_division=0),
        "top_3_accuracy": top_k_accuracy(logits, labels, min(3, logits.shape[1])),
        "top_5_accuracy": top_k_accuracy(logits, labels, min(5, logits.shape[1])),
        "classification_report": classification_report(
            targets, predictions, output_dict=True, zero_division=0
        ),
    }
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CONFUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 12))
    axis.imshow(confusion_matrix(targets, predictions), cmap="Blues")
    axis.set_title("Test Confusion Matrix")
    axis.set_xlabel("Predicted label index")
    axis.set_ylabel("True label index")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=150)
    plt.close(figure)
    print("TEST EVALUATION REPORT")
    print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))
    print(f"Test samples: {len(test_frame)}")
    print(f"Confusion matrix: {CONFUSION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
