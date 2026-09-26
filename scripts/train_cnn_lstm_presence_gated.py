"""Train the CNN-BiLSTM on 228-dimensional presence-gated features."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_cnn_lstm_presence import (  # noqa: E402
    CNNBiLSTMClassifier,
    INPUT_DIM,
    NUM_CLASSES,
    PresenceAwareDataset,
    checkpoint_payload,
    load_config,
    normalize_video_path,
    run_epoch,
    seed_everything,
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features_presence_gated"
FEATURE_MANIFEST = FEATURE_DIR / "manifest.csv"
BEST_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_presence_gated_best.pt"
FINAL_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_presence_gated_final.pt"
HISTORY_PATH = PROJECT_ROOT / "reports" / "cnn_lstm_presence_gated_history.json"
SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = {
    "split",
    "video_path",
    "feature_path",
    "num_frames",
    "feature_dim",
}


def load_gated_data() -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Join split rows to gated-only features and validate every array."""
    if not FEATURE_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Presence-gated feature manifest not found: {FEATURE_MANIFEST}"
        )
    feature_manifest = pd.read_csv(FEATURE_MANIFEST, keep_default_na=False)
    missing_columns = MANIFEST_COLUMNS - set(feature_manifest.columns)
    if missing_columns:
        raise ValueError(
            f"{FEATURE_MANIFEST} is missing columns: {sorted(missing_columns)}"
        )
    if feature_manifest["video_path"].map(normalize_video_path).duplicated().any():
        raise ValueError("Presence-gated manifest has duplicate video_path entries.")

    manifest_by_video = {
        normalize_video_path(row["video_path"]): row
        for _, row in feature_manifest.iterrows()
    }
    split_frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        split_path = SPLIT_DIR / f"{split}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        source = pd.read_csv(split_path, keep_default_na=False)
        required = {"video_path", "label_id"}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"{split_path} is missing columns: {sorted(missing)}")
        if source["video_path"].map(normalize_video_path).duplicated().any():
            raise ValueError(f"{split_path} has duplicate video_path values.")

        records: list[dict[str, str]] = []
        missing_videos: list[str] = []
        for _, split_row in source.iterrows():
            video_key = normalize_video_path(split_row["video_path"])
            manifest_row = manifest_by_video.get(video_key)
            if manifest_row is None:
                missing_videos.append(str(split_row["video_path"]))
                continue
            if str(manifest_row["split"]) != split:
                raise ValueError(
                    f"{split_row['video_path']} is marked as "
                    f"{manifest_row['split']!r}, expected {split!r}."
                )
            feature_path = Path(
                str(manifest_row["feature_path"]).replace("\\", "/")
            )
            if not feature_path.is_absolute():
                feature_path = PROJECT_ROOT / feature_path
            feature_path = feature_path.resolve()
            if feature_path.parent != FEATURE_DIR.resolve():
                raise ValueError(
                    f"Refusing non-gated feature input: {feature_path}"
                )
            if int(manifest_row["feature_dim"]) != INPUT_DIM:
                raise ValueError(
                    f"Expected feature_dim={INPUT_DIM} for {video_key}, "
                    f"got {manifest_row['feature_dim']}."
                )
            try:
                features = np.load(feature_path, allow_pickle=False)
            except (OSError, ValueError, EOFError) as exc:
                raise ValueError(f"Could not load {feature_path}: {exc}") from exc
            if (
                features.ndim != 2
                or features.shape[0] == 0
                or features.shape[1] != INPUT_DIM
            ):
                raise ValueError(
                    f"Expected non-empty (T, {INPUT_DIM}) data in "
                    f"{feature_path}; found {features.shape}."
                )
            if features.dtype != np.float32 or not np.isfinite(features).all():
                raise ValueError(
                    f"Expected finite float32 data in {feature_path}; "
                    f"found dtype {features.dtype}."
                )
            if features.shape[0] != int(manifest_row["num_frames"]):
                raise ValueError(
                    f"Frame count mismatch in {feature_path}: "
                    f"array={features.shape[0]}, "
                    f"manifest={manifest_row['num_frames']}."
                )
            if not np.isin(features[:, 225:228], (0.0, 1.0)).all():
                raise ValueError(f"Non-binary presence flags in {feature_path}.")
            records.append(
                {
                    "video_path": str(split_row["video_path"]),
                    "feature_path": str(feature_path),
                    "label_id": str(split_row["label_id"]),
                }
            )
        if missing_videos:
            raise ValueError(
                f"{len(missing_videos)} {split} videos lack gated features; "
                f"examples: {missing_videos[:5]}"
            )
        split_frames[split] = pd.DataFrame(records)

    mapping = {
        label: index
        for index, label in enumerate(
            sorted(split_frames["train"]["label_id"].astype(str).unique())
        )
    }
    if len(mapping) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(mapping)}.")
    for split, frame in split_frames.items():
        unknown_labels = set(frame["label_id"].astype(str)) - set(mapping)
        if unknown_labels:
            raise ValueError(
                f"{split} has labels absent from train: {sorted(unknown_labels)}"
            )

    split_video_keys = {
        normalize_video_path(video)
        for frame in split_frames.values()
        for video in frame["video_path"]
    }
    if set(manifest_by_video) != split_video_keys:
        raise ValueError(
            "The gated manifest does not exactly match the three split files."
        )
    return split_frames, mapping


def main() -> int:
    output_paths = (BEST_PATH, FINAL_PATH, HISTORY_PATH)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing gated-training outputs: "
            + ", ".join(existing)
        )

    config = load_config()
    seed_everything(int(config["seed"]))
    split_frames, mapping = load_gated_data()
    datasets = {
        split: PresenceAwareDataset(
            frame,
            config["max_frames"],
            mapping,
            training=(split == "train"),
            seed=int(config["seed"]),
            landmark_noise_std=(
                float(config["landmark_noise_std"]) if split == "train" else 0.0
            ),
            frame_dropout_prob=(
                float(config["frame_dropout_prob"]) if split == "train" else 0.0
            ),
        )
        for split, frame in split_frames.items()
    }
    loader_options = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **loader_options),
        "val": DataLoader(datasets["val"], shuffle=False, **loader_options),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_options),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTMClassifier(
        input_dim=INPUT_DIM,
        num_classes=len(mapping),
        cnn_channels_1=config["cnn_channels_1"],
        cnn_channels_2=config["cnn_channels_2"],
        lstm_hidden_size=config["lstm_hidden_size"],
        lstm_layers=config["lstm_layers"],
        cnn_dropout=config["cnn_dropout"],
        lstm_dropout=config["lstm_dropout"],
        classifier_dropout=config["classifier_dropout"],
    ).to(device)

    print("CNN-BiLSTM PRESENCE-GATED TRAINING")
    print("Model architecture:")
    print(model)
    print(
        "Trainable parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}"
    )
    print(f"Input dimension: {INPUT_DIM}")
    print(f"Train samples: {len(split_frames['train'])}")
    print(f"Validation samples: {len(split_frames['val'])}")
    print(f"Test samples: {len(split_frames['test'])}")
    print(f"Classes: {len(mapping)}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {'enabled' if device.type == 'cuda' else 'disabled (CPU)'}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: dict[str, Any] = {
        "model_config": {**config, "input_dim": INPUT_DIM},
        "dataset_sizes": {
            split: len(frame) for split, frame in split_frames.items()
        },
        "history": {
            key: []
            for key in (
                "train_loss",
                "train_accuracy",
                "train_macro_f1",
                "train_weighted_f1",
                "train_top3",
                "train_top5",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "val_weighted_f1",
                "val_top3",
                "val_top5",
                "learning_rate",
            )
        },
    }
    best_macro_f1 = -1.0
    best_validation: dict[str, float] | None = None
    best_epoch = 0
    stale_epochs = 0
    epoch = 0
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config["epochs"]) + 1):
        datasets["train"].set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            len(mapping),
            optimizer,
            scaler,
            float(config["gradient_clip"]),
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, loaders["val"], criterion, device, len(mapping)
            )
        values = {
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        for key, value in values.items():
            history["history"][key].append(value)
        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} "
            f"weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"top3={val_metrics['top3']:.4f} "
            f"top5={val_metrics['top5']:.4f}"
        )
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_validation = val_metrics.copy()
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                checkpoint_payload(
                    model, optimizer, epoch, best_macro_f1, mapping, config
                ),
                BEST_PATH,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                print(f"Early stopping after epoch {epoch}.")
                break

    if best_validation is None:
        raise RuntimeError("Training ended without a best validation checkpoint.")
    torch.save(
        checkpoint_payload(
            model, optimizer, epoch, best_macro_f1, mapping, config
        ),
        FINAL_PATH,
    )
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history["best_epoch"] = best_epoch
    history["best_validation_metrics"] = best_validation
    HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    best_checkpoint = torch.load(
        BEST_PATH, map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    with torch.no_grad():
        test_metrics = run_epoch(
            model, loaders["test"], criterion, device, len(mapping)
        )
    history["test_metrics_at_best_validation_checkpoint"] = test_metrics
    HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print("\nBEST VALIDATION RESULTS")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_validation['macro_f1']:.4f}")
    print(f"Best validation accuracy: {best_validation['accuracy']:.4f}")
    print(f"Best validation top-3: {best_validation['top3']:.4f}")
    print(f"Best validation top-5: {best_validation['top5']:.4f}")
    print(
        "Test metrics at best validation checkpoint: "
        f"loss={test_metrics['loss']:.4f}, "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}, "
        f"weighted_f1={test_metrics['weighted_f1']:.4f}, "
        f"top3={test_metrics['top3']:.4f}, "
        f"top5={test_metrics['top5']:.4f}"
    )
    print(f"Best checkpoint: {BEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Final checkpoint: {FINAL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"History: {HISTORY_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
