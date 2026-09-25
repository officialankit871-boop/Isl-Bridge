"""Train the CNN + BiLSTM ISL sentence-level recognition model."""

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

from cnn_lstm_model import CNNBiLSTMClassifier  # noqa: E402
from dataset import (  # noqa: E402
    LandmarkDataset,
    class_mapping,
    resolve_path,
    training_class_weights,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "cnn_lstm_config.yaml"
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "val.csv"
BEST_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_best.pt"
FINAL_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_final.pt"
HISTORY_PATH = PROJECT_ROOT / "reports" / "cnn_lstm_training_history.json"
CURVE_PATH = PROJECT_ROOT / "plots" / "cnn_lstm_training_curves.png"
INPUT_DIM = 225
NUM_CLASSES = 101


def seed_everything(seed: int) -> None:
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config() -> dict[str, Any]:
    """Read the dedicated CNN + BiLSTM configuration."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_split(csv_path: Path) -> pd.DataFrame:
    """Validate metadata and every referenced landmark file before training."""
    frame = pd.read_csv(csv_path, keep_default_na=False)
    required = {"landmark_path", "feature_dim", "label_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    if not (frame["feature_dim"].astype(int) == INPUT_DIM).all():
        raise ValueError(f"Every feature_dim in {csv_path} must equal {INPUT_DIM}.")

    for raw_path in frame["landmark_path"]:
        path = resolve_path(str(raw_path))
        if not path.is_file():
            raise FileNotFoundError(f"Landmark file does not exist: {path}")
        try:
            values = np.load(path, allow_pickle=False)
            values = values.astype(np.float32, copy=False)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Could not load landmark file {path}: {exc}") from exc
        if values.ndim != 2 or values.shape[1] != INPUT_DIM:
            raise ValueError(f"Expected (T, {INPUT_DIM}) in {path}, got {values.shape}")
        if values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError(f"Invalid or empty landmark values in {path}")
    return frame


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    """Count samples whose target appears in the top-k predictions."""
    return int(
        logits.topk(min(k, logits.shape[1]), dim=1).indices.eq(
            labels.unsqueeze(1)
        ).any(dim=1).sum()
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
    """Run one train or validation epoch."""
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    top3 = 0
    top5 = 0
    total = 0

    for batch in loader:
        features = batch["features"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["label"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = model(features, padding_mask)
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
        top3 += top_k_accuracy(logits, labels, 3)
        top5 += top_k_accuracy(logits, labels, 5)
        total += labels.size(0)

    labels_range = list(range(num_classes))
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets, predictions, labels=labels_range, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            targets,
            predictions,
            labels=labels_range,
            average="weighted",
            zero_division=0,
        ),
        "top3": top3 / max(1, total),
        "top5": top5 / max(1, total),
    }


def save_curves(history: dict[str, list[float]]) -> None:
    """Save loss and macro-F1 training curves."""
    CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_macro_f1"], label="train")
    axes[1].plot(history["val_macro_f1"], label="validation")
    axes[1].set_title("Macro F1")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(CURVE_PATH, dpi=150)
    plt.close(figure)


def main() -> int:
    """Validate data, train, checkpoint, and record metrics."""
    config = load_config()
    seed_everything(int(config["seed"]))
    train_frame = validate_split(TRAIN_CSV)
    val_frame = validate_split(VAL_CSV)
    mapping = class_mapping(train_frame)
    if len(mapping) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(mapping)}.")

    train_dataset = LandmarkDataset(
        TRAIN_CSV,
        config["max_frames"],
        mapping,
        training=True,
        seed=config["seed"],
        landmark_noise_std=config["landmark_noise_std"],
        frame_dropout_prob=config["frame_dropout_prob"],
    )
    val_dataset = LandmarkDataset(
        VAL_CSV,
        config["max_frames"],
        mapping,
        training=False,
        seed=config["seed"],
    )
    loader_options = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Train samples: {len(train_frame)}")
    print(f"Validation samples: {len(val_frame)}")
    print(f"Number of classes: {len(mapping)}")
    print(f"Feature dimension: {INPUT_DIM}")
    print(f"Maximum frames: {config['max_frames']}")

    model = CNNBiLSTMClassifier(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        cnn_channels_1=config["cnn_channels_1"],
        cnn_channels_2=config["cnn_channels_2"],
        lstm_hidden_size=config["lstm_hidden_size"],
        lstm_layers=config["lstm_layers"],
        cnn_dropout=config["cnn_dropout"],
        lstm_dropout=config["lstm_dropout"],
        classifier_dropout=config["classifier_dropout"],
    ).to(device)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Input shape: (B, {config['max_frames']}, {INPUT_DIM})")
    print(
          f"CNN output: (B, {config['max_frames']}, "
          f"{config['cnn_channels_2']})"
    )
    print(
          f"BiLSTM output: (B, {config['max_frames']}, "
          f"{config['lstm_hidden_size'] * 2})"
    )
    print(f"Final output: (B, {NUM_CLASSES})")

    class_weights = training_class_weights(
        train_frame, mapping, config["max_class_weight"]
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights if config["use_class_weights"] else None,
        label_smoothing=config["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: dict[str, list[float]] = {
        key: []
        for key in (
            "train_loss",
            "train_accuracy",
            "train_macro_f1",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "val_weighted_f1",
            "val_top3",
            "val_top5",
            "learning_rate",
        )
    }

    best_f1 = -1.0
    stale_epochs = 0
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            NUM_CLASSES,
            optimizer,
            scaler,
            config["gradient_clip"],
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, NUM_CLASSES)
        learning_rate = optimizer.param_groups[0]["lr"]
        values = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_top3": val_metrics["top3"],
            "val_top5": val_metrics["top5"],
            "learning_rate": learning_rate,
        }
        for key, value in values.items():
            history[key].append(value)
        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Accuracy: {train_metrics['accuracy']:.4f} | "
            f"Train Macro F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Weighted F1: {val_metrics['weighted_f1']:.4f} | "
            f"Val Top-3: {val_metrics['top3']:.4f} | "
            f"Val Top-5: {val_metrics['top5']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "class_to_index": mapping,
                    "model_config": config,
                },
                BEST_PATH,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                print(f"Early stopping after epoch {epoch}.")
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_macro_f1": best_f1,
            "class_to_index": mapping,
            "model_config": config,
        },
        FINAL_PATH,
    )
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_curves(history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
