"""Train the existing CNN-BiLSTM on presence-aware 228-feature sequences."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cnn_lstm_model import CNNBiLSTMClassifier  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "cnn_lstm_config.yaml"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features_presence_aware"
FEATURE_MANIFEST = FEATURE_DIR / "manifest.csv"
BEST_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_presence_best.pt"
FINAL_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_presence_final.pt"
HISTORY_PATH = PROJECT_ROOT / "reports" / "cnn_lstm_presence_history.json"
INPUT_DIM = 228
NUM_CLASSES = 101
SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = {"split", "video_path", "feature_path", "num_frames", "feature_dim"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "seed": 42,
        "max_frames": 128,
        "batch_size": 16,
        "epochs": 60,
        "learning_rate": 0.0001,
        "weight_decay": 0.001,
        "cnn_channels_1": 64,
        "cnn_channels_2": 128,
        "lstm_hidden_size": 64,
        "lstm_layers": 2,
        "cnn_dropout": 0.25,
        "lstm_dropout": 0.30,
        "classifier_dropout": 0.50,
        "patience": 12,
        "gradient_clip": 1.0,
        "use_class_weights": False,
        "label_smoothing": 0.05,
        "landmark_noise_std": 0.01,
        "frame_dropout_prob": 0.05,
        "num_workers": 0,
    }
    for key, expected in required.items():
        if key not in config:
            raise ValueError(f"Training config is missing {key!r}.")
        actual = config[key]
        if isinstance(expected, float):
            matches = float(actual) == expected
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Controlled comparison requires {key}={expected!r}; "
                f"found {actual!r} in {CONFIG_PATH}."
            )
    return config


def normalize_video_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def load_training_data() -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    if not FEATURE_MANIFEST.is_file():
        raise FileNotFoundError(f"Presence-aware feature manifest not found: {FEATURE_MANIFEST}")
    feature_manifest = pd.read_csv(FEATURE_MANIFEST, keep_default_na=False)
    missing_columns = MANIFEST_COLUMNS - set(feature_manifest.columns)
    if missing_columns:
        raise ValueError(
            f"{FEATURE_MANIFEST} is missing columns: {sorted(missing_columns)}"
        )
    if feature_manifest["video_path"].map(normalize_video_path).duplicated().any():
        raise ValueError("Presence-aware manifest contains duplicate video_path keys.")

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
            raise ValueError(f"{split_path} contains duplicate video_path values.")

        records: list[dict[str, Any]] = []
        missing_videos: list[str] = []
        for _, row in source.iterrows():
            normalized_path = normalize_video_path(row["video_path"])
            entry = manifest_by_video.get(normalized_path)
            if entry is None:
                missing_videos.append(str(row["video_path"]))
                continue
            if str(entry["split"]) != split:
                raise ValueError(
                    f"Feature manifest places {row['video_path']} in split "
                    f"{entry['split']!r}, expected {split!r}."
                )
            feature_path = Path(str(entry["feature_path"]).replace("\\", "/"))
            if not feature_path.is_absolute():
                feature_path = PROJECT_ROOT / feature_path
            feature_path = feature_path.resolve()
            if feature_path.parent != FEATURE_DIR.resolve():
                raise ValueError(
                    f"Refusing non-presence-aware feature path: {feature_path}"
                )
            if int(entry["feature_dim"]) != INPUT_DIM:
                raise ValueError(
                    f"Feature manifest dimension for {row['video_path']} is "
                    f"{entry['feature_dim']}, expected {INPUT_DIM}."
                )
            try:
                features = np.load(feature_path, allow_pickle=False)
            except (OSError, ValueError, EOFError) as exc:
                raise ValueError(
                    f"Could not load presence-aware feature file {feature_path}: {exc}"
                ) from exc
            if (
                features.ndim != 2
                or features.shape[0] == 0
                or features.shape[1] != INPUT_DIM
            ):
                raise ValueError(
                    f"Expected non-empty (T, {INPUT_DIM}) features in "
                    f"{feature_path}; got {features.shape}."
                )
            if features.dtype != np.float32 or not np.isfinite(features).all():
                raise ValueError(
                    f"Expected finite float32 features in {feature_path}; "
                    f"got dtype={features.dtype}."
                )
            if features.shape[0] != int(entry["num_frames"]):
                raise ValueError(
                    f"Frame count mismatch for {feature_path}: "
                    f"array={features.shape[0]}, manifest={entry['num_frames']}."
                )
            if not np.isin(features[:, 225:228], (0.0, 1.0)).all():
                raise ValueError(
                    f"Presence columns are not binary in {feature_path}."
                )
            records.append(
                {
                    "video_path": str(row["video_path"]),
                    "feature_path": str(feature_path),
                    "label_id": str(row["label_id"]),
                }
            )
        if missing_videos:
            raise ValueError(
                f"{len(missing_videos)} {split} rows are missing presence-aware "
                f"features; examples: {missing_videos[:5]}"
            )
        split_frames[split] = pd.DataFrame(records)

    mapping = {
        label: index
        for index, label in enumerate(
            sorted(split_frames["train"]["label_id"].astype(str).unique())
        )
    }
    if len(mapping) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} training classes, found {len(mapping)}.")
    for split, frame in split_frames.items():
        unknown = set(frame["label_id"].astype(str)) - set(mapping)
        if unknown:
            raise ValueError(f"{split} contains labels not present in train: {sorted(unknown)}")
    manifest_videos = set(manifest_by_video)
    split_videos = {
        normalize_video_path(path)
        for frame in split_frames.values()
        for path in frame["video_path"]
    }
    if manifest_videos != split_videos:
        raise ValueError(
            "Presence-aware manifest has entries outside the train/val/test "
            "splits or does not cover all split videos."
        )
    return split_frames, mapping


def prepare_sequence(
    sequence: np.ndarray,
    max_frames: int,
    training: bool,
    rng: np.random.Generator | None,
    landmark_noise_std: float,
    frame_dropout_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sequence.ndim != 2 or sequence.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected (T, {INPUT_DIM}), got {sequence.shape}.")
    sequence = sequence.astype(np.float32, copy=False)
    if sequence.shape[0] > max_frames:
        start = (
            int(rng.integers(0, sequence.shape[0] - max_frames + 1))
            if training and rng is not None
            else 0
        )
        sequence = sequence[start : start + max_frames]
    valid_length = min(sequence.shape[0], max_frames)
    output = np.zeros((max_frames, INPUT_DIM), dtype=np.float32)
    output[:valid_length] = sequence[:valid_length]
    padding_mask = np.ones(max_frames, dtype=bool)
    padding_mask[:valid_length] = False
    if training and rng is not None and valid_length:
        valid = output[:valid_length]
        if landmark_noise_std > 0:
            valid += rng.normal(
                0.0, landmark_noise_std, size=valid.shape
            ).astype(np.float32)
        if frame_dropout_prob > 0:
            dropped = rng.random(valid_length) < frame_dropout_prob
            valid[dropped] = 0.0
    return output, padding_mask


class PresenceAwareDataset(Dataset[dict[str, torch.Tensor]]):
    """Load 228-feature sequences with the existing temporal augmentation."""

    def __init__(
        self,
        frame: pd.DataFrame,
        max_frames: int,
        class_to_index: dict[str, int],
        training: bool,
        seed: int,
        landmark_noise_std: float,
        frame_dropout_prob: float,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.max_frames = max_frames
        self.class_to_index = class_to_index
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.landmark_noise_std = landmark_noise_std
        self.frame_dropout_prob = frame_dropout_prob

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        path = Path(str(row["feature_path"]))
        try:
            sequence = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise RuntimeError(f"Could not load feature file {path}: {exc}") from exc
        if sequence.dtype != np.float32 or not np.isfinite(sequence).all():
            raise ValueError(f"Invalid feature values or dtype in {path}.")
        rng = (
            np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            if self.training
            else None
        )
        features, padding_mask = prepare_sequence(
            sequence,
            self.max_frames,
            self.training,
            rng,
            self.landmark_noise_std,
            self.frame_dropout_prob,
        )
        label_id = str(row["label_id"])
        return {
            "features": torch.from_numpy(features),
            "padding_mask": torch.from_numpy(padding_mask),
            "label": torch.tensor(self.class_to_index[label_id], dtype=torch.long),
        }


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    return int(
        logits.topk(min(k, logits.shape[1]), dim=1)
        .indices.eq(labels.unsqueeze(1))
        .any(dim=1)
        .sum()
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
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    top3 = 0
    top5 = 0
    total = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
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
        targets.extend(labels.detach().cpu().tolist())
        top3 += top_k_accuracy(logits, labels, 3)
        top5 += top_k_accuracy(logits, labels, 5)
        total += int(labels.size(0))
    labels_range = list(range(num_classes))
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels_range,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels_range,
                average="weighted",
                zero_division=0,
            )
        ),
        "top3": top3 / max(1, total),
        "top5": top5 / max(1, total),
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_macro_f1: float,
    class_to_index: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_macro_f1": best_macro_f1,
        "class_to_index": class_to_index,
        "model_config": {**config, "input_dim": INPUT_DIM},
    }


def main() -> int:
    if any(path.exists() for path in (BEST_PATH, FINAL_PATH, HISTORY_PATH)):
        existing = [str(path) for path in (BEST_PATH, FINAL_PATH, HISTORY_PATH) if path.exists()]
        raise FileExistsError(
            "Refusing to overwrite existing presence-training outputs: "
            + ", ".join(existing)
        )
    config = load_config()
    seed_everything(int(config["seed"]))
    split_frames, mapping = load_training_data()

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
    train_loader = DataLoader(
        datasets["train"], shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        datasets["val"], shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        datasets["test"], shuffle=False, **loader_options
    )

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
    print("CNN-BiLSTM PRESENCE-AWARE TRAINING")
    print("Model architecture:")
    print(model)
    print(
        f"Trainable parameters: "
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
            train_loader,
            criterion,
            device,
            len(mapping),
            optimizer,
            scaler,
            float(config["gradient_clip"]),
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, criterion, device, len(mapping)
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
                    model,
                    optimizer,
                    epoch,
                    best_macro_f1,
                    mapping,
                    config,
                ),
                BEST_PATH,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                print(f"Early stopping after epoch {epoch}.")
                break

    if best_validation is None:
        raise RuntimeError("Training completed without producing a best checkpoint.")
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            epoch,
            best_macro_f1,
            mapping,
            config,
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

    best_checkpoint = torch.load(BEST_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    with torch.no_grad():
        test_metrics = run_epoch(
            model, test_loader, criterion, device, len(mapping)
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
        f"top3={test_metrics['top3']:.4f}, top5={test_metrics['top5']:.4f}"
    )
    print(f"Best checkpoint: {BEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Final checkpoint: {FINAL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"History: {HISTORY_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
