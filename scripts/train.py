"""Train the ISL sentence-level temporal Transformer."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import LandmarkDataset, class_mapping, training_class_weights  # noqa: E402
from model import TemporalTransformerClassifier  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
FINAL_PATH = PROJECT_ROOT / "checkpoints" / "final_model.pt"
HISTORY_PATH = PROJECT_ROOT / "reports" / "training_history.json"
CURVE_PATH = PROJECT_ROOT / "plots" / "training_curves.png"


def seed_everything(seed: int) -> None:
    """Set deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config() -> dict[str, Any]:
    """Load YAML training configuration."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def device_info() -> torch.device:
    """Select CUDA when available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def make_loaders(config: dict[str, Any], mapping: dict[str, int]) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    """Build train/validation loaders and return the training frame."""
    train_frame = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "train.csv")
    train_dataset = LandmarkDataset(
        PROJECT_ROOT / "data" / "splits" / "train.csv",
        config["max_frames"], mapping, training=True, seed=config["seed"],
        landmark_noise_std=config["landmark_noise_std"],
        frame_dropout_prob=config["frame_dropout_prob"],
    )
    val_dataset = LandmarkDataset(
        PROJECT_ROOT / "data" / "splits" / "val.csv",
        config["max_frames"], mapping, training=False, seed=config["seed"],
    )
    common = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
        train_frame,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    """Run one training or validation epoch and calculate aggregate metrics."""
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    top3_correct = 0
    top5_correct = 0
    total_examples = 0
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["padding_mask"].to(device)
        labels = batch["label"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = model(features, mask)
            loss = criterion(logits, labels)
        if training and optimizer is not None:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.cpu().tolist())
        top3_correct += int(
            logits.topk(min(3, logits.shape[1]), dim=1).indices.eq(
                labels.unsqueeze(1)
            ).any(dim=1).sum()
        )
        top5_correct += int(
            logits.topk(min(5, logits.shape[1]), dim=1).indices.eq(
                labels.unsqueeze(1)
            ).any(dim=1).sum()
        )
        total_examples += labels.size(0)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets, predictions, labels=list(range(num_classes)),
            average="macro", zero_division=0,
        ),
        "weighted_f1": f1_score(
            targets, predictions, labels=list(range(num_classes)),
            average="weighted", zero_division=0,
        ),
        "top3": top3_correct / max(1, total_examples),
        "top5": top5_correct / max(1, total_examples),
    }


def plot_history(history: dict[str, list[float]]) -> None:
    """Save loss and metric curves."""
    CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_macro_f1"], label="train macro F1")
    axes[1].plot(history["val_macro_f1"], label="val macro F1")
    axes[1].set_title("Macro F1")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(CURVE_PATH, dpi=150)
    plt.close(figure)


def main() -> int:
    """Train, checkpoint, and record the best Transformer model."""
    config = load_config()
    seed_everything(int(config["seed"]))
    device = device_info()
    train_frame = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "train.csv")
    mapping = class_mapping(train_frame)
    train_loader, val_loader, train_frame = make_loaders(config, mapping)
    model = TemporalTransformerClassifier(
        input_dim=225, num_classes=len(mapping), max_frames=config["max_frames"],
        d_model=config["d_model"], nhead=config["nhead"],
        num_layers=config["num_layers"], dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
    ).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Dataset sizes: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
    print(f"Number of classes: {len(mapping)}")
    weights = training_class_weights(
        train_frame, mapping, config["max_class_weight"]
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights if config["use_class_weights"] else None,
        label_smoothing=config["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, config["patience"] // 3)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: dict[str, list[float]] = {key: [] for key in (
        "train_loss", "train_accuracy", "train_macro_f1", "val_loss",
        "val_accuracy", "val_macro_f1", "val_weighted_f1", "val_top3", "learning_rate",
        "val_top5",
    )}
    best_f1 = -1.0
    stale_epochs = 0
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, len(mapping),
            optimizer, scaler, config["gradient_clip"]
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, len(mapping))
        scheduler.step(val_metrics["macro_f1"])
        learning_rate = optimizer.param_groups[0]["lr"]
        for key, value in {
            "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"], "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"], "val_top3": val_metrics["top3"],
            "val_top5": val_metrics["top5"],
            "learning_rate": learning_rate,
        }.items():
            history[key].append(value)
        print(
            f"Epoch {epoch}/{config['epochs']}\n"
            f"Train Loss: {train_metrics['loss']:.4f} | Train Accuracy: {train_metrics['accuracy']:.4f} | "
            f"Train Macro F1: {train_metrics['macro_f1']:.4f}\n"
            f"Val Loss: {val_metrics['loss']:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | Val Weighted F1: {val_metrics['weighted_f1']:.4f} | "
            f"Val Top-3: {val_metrics['top3']:.4f} | Val Top-5: {val_metrics['top5']:.4f} | "
            f"Learning Rate: {learning_rate:.6g}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_macro_f1": best_f1,
                "class_to_index": mapping,
                "model_config": config,
            }, CHECKPOINT_PATH)
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                print(f"Early stopping after epoch {epoch}.")
                break
    torch.save({"model_state_dict": model.state_dict(), "class_to_index": mapping, "model_config": config}, FINAL_PATH)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
