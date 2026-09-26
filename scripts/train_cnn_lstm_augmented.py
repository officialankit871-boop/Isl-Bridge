"""Train the baseline CNN-BiLSTM with train-only realistic augmentation."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import sys
import tempfile
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

CONFIG_PATH = PROJECT_ROOT / "configs" / "cnn_lstm_augmented_config.yaml"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
LANDMARK_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
BASELINE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "cnn_lstm_best.pt"
BASELINE_ANALYSIS = PROJECT_ROOT / "reports" / "model_error_analysis.json"
BEST_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_augmented_best.pt"
FINAL_PATH = PROJECT_ROOT / "checkpoints" / "cnn_lstm_augmented_final.pt"
HISTORY_PATH = PROJECT_ROOT / "reports" / "cnn_lstm_augmented_history.json"
RESULTS_PATH = PROJECT_ROOT / "reports" / "cnn_lstm_augmented_results.json"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "cnn_lstm_augmented_comparison.json"
SPLITS = ("train", "val", "test")
INPUT_DIM = 225
NUM_CLASSES = 101
SPLIT_COLUMNS = {"landmark_path", "video_path", "label_id", "feature_dim"}
OUTPUT_PATHS = (
    BEST_PATH,
    FINAL_PATH,
    HISTORY_PATH,
    RESULTS_PATH,
    COMPARISON_PATH,
)
COMPARISON_METRICS = (
    ("accuracy", "accuracy"),
    ("macro_f1", "macro_f1"),
    ("weighted_f1", "weighted_f1"),
    ("top3", "top3"),
    ("top5", "top5"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the controlled CNN-BiLSTM augmentation experiment."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run an isolated 4/2/2-sample training/checkpoint smoke test.",
    )
    modes.add_argument(
    "--generate-comparison",
    action="store_true",
    help="Generate the baseline comparison from existing experiment results without training.",
    )
    modes.add_argument(
        "--check-results",
        action="store_true",
        help="Validate the saved experiment results without training.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_config(config: dict[str, Any]) -> None:
    expected: dict[str, Any] = {
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
        "max_class_weight": 3.0,
        "label_smoothing": 0.05,
        "num_workers": 0,
    }
    for key, expected_value in expected.items():
        if key not in config:
            raise ValueError(f"{CONFIG_PATH} is missing {key!r}.")
        actual_value = config[key]
        matches = (
            float(actual_value) == expected_value
            if isinstance(expected_value, float)
            else actual_value == expected_value
        )
        if not matches:
            raise ValueError(
                f"Controlled experiment requires {key}={expected_value!r}; "
                f"found {actual_value!r}."
            )
    augmentation = config.get("augmentation")
    if not isinstance(augmentation, dict) or not isinstance(
        augmentation.get("enabled"), bool
    ):
        raise ValueError("Config must contain a boolean augmentation.enabled.")
    if augmentation["enabled"] is not True:
        raise ValueError("The controlled experiment requires augmentation.enabled=true.")
    expected_augmentation = {
        "temporal_crop": {
            "enabled": True,
            "probability": 0.5,
        },
        "temporal_scale": {
            "enabled": True,
            "probability": 0.3,
            "min_scale": 0.90,
            "max_scale": 1.10,
        },
        "landmark_noise": {
            "enabled": True,
            "probability": 0.3,
            "std": 0.005,
        },
        "frame_dropout": {
            "enabled": True,
            "probability": 0.3,
            "frame_probability": 0.05,
        },
    }
    if augmentation != {"enabled": augmentation["enabled"], **expected_augmentation}:
        raise ValueError(
            "Augmentation configuration differs from the controlled defaults."
        )


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {CONFIG_PATH}")
    validate_config(config)
    return config


def normalize_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def resolve_landmark_path(value: Any) -> Path:
    path = Path(str(value).replace("\\", "/"))
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if resolved.parent != LANDMARK_DIR.resolve() or resolved.suffix.lower() != ".npy":
        raise ValueError(
            f"Split path is not a direct original-landmark file: {value}"
        )
    return resolved


def load_and_validate_splits() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    video_owners: dict[str, str] = {}
    path_owners: dict[str, str] = {}
    for split in SPLITS:
        csv_path = SPLIT_DIR / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Split file not found: {csv_path}")
        frame = pd.read_csv(csv_path, keep_default_na=False)
        missing = SPLIT_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{csv_path} is empty.")
        if not (frame["feature_dim"].astype(int) == INPUT_DIM).all():
            raise ValueError(
                f"All feature_dim values in {csv_path} must equal {INPUT_DIM}."
            )
        if frame["label_id"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{csv_path} contains empty labels.")
        for _, row in frame.iterrows():
            video_key = normalize_path(row["video_path"])
            path = resolve_landmark_path(row["landmark_path"])
            path_key = str(path).casefold()
            if video_key in video_owners:
                raise ValueError(
                    f"Video path overlaps {video_owners[video_key]} and {split}: "
                    f"{row['video_path']}"
                )
            if path_key in path_owners:
                raise ValueError(
                    f"Landmark path overlaps {path_owners[path_key]} and {split}: {path}"
                )
            video_owners[video_key] = split
            path_owners[path_key] = split
            if not path.is_file():
                raise FileNotFoundError(f"Landmark file does not exist: {path}")
            try:
                values = np.load(path, allow_pickle=False)
            except (OSError, ValueError, TypeError, EOFError) as exc:
                raise ValueError(f"Could not load landmark file {path}: {exc}") from exc
            if (
                values.ndim != 2
                or values.shape[0] == 0
                or values.shape[1] != INPUT_DIM
            ):
                raise ValueError(
                    f"Expected non-empty (T, {INPUT_DIM}) in {path}; "
                    f"found {values.shape}."
                )
            if values.dtype != np.float32:
                raise ValueError(
                    f"Expected float32 landmarks in {path}; found {values.dtype}."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Landmark file contains NaN/Inf: {path}")
        frames[split] = frame.reset_index(drop=True)

    labels = set(frames["train"]["label_id"].astype(str))
    if len(labels) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} train labels; found {len(labels)}.")
    for split in ("val", "test"):
        unknown = set(frames[split]["label_id"].astype(str)) - labels
        if unknown:
            raise ValueError(f"{split} contains unknown train labels: {sorted(unknown)}")
    return frames


def class_mapping(train_frame: pd.DataFrame) -> dict[str, int]:
    labels = sorted(train_frame["label_id"].astype(str).unique())
    return {label: index for index, label in enumerate(labels)}


def interpolate_time(sequence: np.ndarray, scale: float) -> np.ndarray:
    if sequence.shape[0] <= 1 or scale == 1.0:
        return sequence
    target_length = max(1, int(round(sequence.shape[0] * scale)))
    if target_length == sequence.shape[0]:
        return sequence
    source_positions = np.linspace(
        0.0, sequence.shape[0] - 1, num=target_length, dtype=np.float64
    )
    lower = np.floor(source_positions).astype(np.int64)
    upper = np.minimum(lower + 1, sequence.shape[0] - 1)
    fraction = (source_positions - lower).astype(np.float32)[:, None]
    return (
        sequence[lower] * (1.0 - fraction) + sequence[upper] * fraction
    ).astype(np.float32, copy=False)


def apply_temporal_augmentations(
    sequence: np.ndarray,
    max_frames: int,
    augmentation: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Apply temporal scaling/crop, then zero-pad to max_frames."""
    transformed = sequence
    if augmentation["enabled"]:
        scaling = augmentation["temporal_scale"]
        if (
            scaling["enabled"]
            and rng.random() < float(scaling["probability"])
        ):
            scale = float(
                rng.uniform(float(scaling["min_scale"]), float(scaling["max_scale"]))
            )
            transformed = interpolate_time(transformed, scale)

    if transformed.shape[0] > max_frames:
        crop = augmentation["temporal_crop"]
        if (
            augmentation["enabled"]
            and crop["enabled"]
            and rng.random() < float(crop["probability"])
        ):
            start = int(rng.integers(0, transformed.shape[0] - max_frames + 1))
        else:
            start = 0
        transformed = transformed[start : start + max_frames]

    valid_length = int(transformed.shape[0])
    output = np.zeros((max_frames, INPUT_DIM), dtype=np.float32)
    output[:valid_length] = transformed
    return output, valid_length


def apply_landmark_noise_and_dropout(
    features: np.ndarray,
    valid_length: int,
    augmentation: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    if valid_length <= 0 or not augmentation["enabled"]:
        return
    noise = augmentation["landmark_noise"]
    if noise["enabled"] and rng.random() < float(noise["probability"]):
        features[:valid_length] += rng.normal(
            0.0,
            float(noise["std"]),
            size=features[:valid_length].shape,
        ).astype(np.float32)

    dropout = augmentation["frame_dropout"]
    if not dropout["enabled"] or rng.random() >= float(dropout["probability"]):
        return
    dropped = rng.random(valid_length) < float(dropout["frame_probability"])
    if dropped.all():
        dropped[int(rng.integers(0, valid_length))] = False
    features[np.flatnonzero(dropped), :] = 0.0


class LandmarkSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Original landmark dataset with epoch-varying train-only augmentation."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_index: dict[str, int],
        max_frames: int,
        seed: int,
        training: bool,
        augmentation: dict[str, Any],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.max_frames = max_frames
        self.seed = seed
        self.training = training
        self.augmentation = augmentation
        self.epoch = 0
        unknown = set(self.frame["label_id"].astype(str)) - set(class_to_index)
        if unknown:
            raise ValueError(f"Dataset has unknown class labels: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        path = resolve_landmark_path(row["landmark_path"])
        sequence = np.load(path, allow_pickle=False)
        if sequence.dtype != np.float32 or not np.isfinite(sequence).all():
            raise ValueError(f"Invalid dtype or non-finite data in {path}.")
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, index])
        )
        training = self.training
        augmentation = self.augmentation if training else NO_AUGMENTATION
        features, valid_length = apply_temporal_augmentations(
            sequence,
            self.max_frames,
            augmentation,
            rng,
        )
        if training:
            apply_landmark_noise_and_dropout(
                features,
                valid_length,
                augmentation,
                rng,
            )
        padding_mask = np.ones(self.max_frames, dtype=np.bool_)
        padding_mask[:valid_length] = False
        return {
            "features": torch.from_numpy(features),
            "padding_mask": torch.from_numpy(padding_mask),
            "label": torch.tensor(
                self.class_to_index[str(row["label_id"])],
                dtype=torch.long,
            ),
        }


NO_AUGMENTATION: dict[str, Any] = {
    "enabled": False,
    "temporal_crop": {"enabled": False, "probability": 0.0},
    "temporal_scale": {
        "enabled": False,
        "probability": 0.0,
        "min_scale": 1.0,
        "max_scale": 1.0,
    },
    "landmark_noise": {"enabled": False, "probability": 0.0, "std": 0.0},
    "frame_dropout": {
        "enabled": False,
        "probability": 0.0,
        "frame_probability": 0.0,
    },
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
        targets.extend(labels.detach().cpu().tolist())
        top3 += top_k_accuracy(logits, labels, 3)
        top5 += top_k_accuracy(logits, labels, 5)
        total += int(labels.size(0))

    label_range = list(range(num_classes))
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=label_range,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                targets,
                predictions,
                labels=label_range,
                average="weighted",
                zero_division=0,
            )
        ),
        "top3": top3 / max(total, 1),
        "top5": top5 / max(total, 1),
    }


def build_model(config: dict[str, Any], device: torch.device) -> CNNBiLSTMClassifier:
    return CNNBiLSTMClassifier(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        cnn_channels_1=int(config["cnn_channels_1"]),
        cnn_channels_2=int(config["cnn_channels_2"]),
        lstm_hidden_size=int(config["lstm_hidden_size"]),
        lstm_layers=int(config["lstm_layers"]),
        cnn_dropout=float(config["cnn_dropout"]),
        lstm_dropout=float(config["lstm_dropout"]),
        classifier_dropout=float(config["classifier_dropout"]),
    ).to(device)


def make_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_macro_f1: float,
    mapping: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "class_to_index": mapping,
        "model_config": {
            key: value for key, value in config.items() if key != "augmentation"
        }
        | {"input_dim": INPUT_DIM},
        "augmentation_config": copy.deepcopy(config["augmentation"]),
    }


def baseline_metrics() -> dict[str, dict[str, float]]:
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}"
        )

    if not BASELINE_ANALYSIS.is_file():
        raise FileNotFoundError(
            f"Baseline metric report not found: {BASELINE_ANALYSIS}. "
            "Run scripts/error_analysis_cnn_lstm.py first."
        )

    report = json.loads(
        BASELINE_ANALYSIS.read_text(encoding="utf-8")
    )

    original = report.get("models", {}).get("original")

    if not isinstance(original, dict):
        raise ValueError(
            "Baseline report does not contain original model metrics."
        )

    reported_checkpoint = str(
        original.get("checkpoint", "")
    ).replace("\\", "/")

    if reported_checkpoint != "checkpoints/cnn_lstm_best.pt":
        raise ValueError(
            f"Baseline report references unexpected checkpoint: "
            f"{reported_checkpoint}"
        )

    metrics: dict[str, dict[str, float]] = {}

    for split in ("val", "test"):
        summary = (
            original
            .get("splits", {})
            .get(split, {})
            .get("summary")
        )

        if not isinstance(summary, dict):
            raise ValueError(
                f"Baseline report is missing {split} metrics."
            )

        metrics[split] = {
            "accuracy": float(summary["accuracy"]),
            "macro_f1": float(summary["macro_f1"]),
            "weighted_f1": float(summary["weighted_f1"]),
            "top3": float(summary["top3_accuracy"]),
            "top5": float(summary["top5_accuracy"]),
       }
 
    return metrics

def difference_report(
    baseline: dict[str, dict[str, float]],
    augmented_val: dict[str, float],
    augmented_test: dict[str, float],
) -> dict[str, Any]:
    augmented = {
        "val": {
            key: float(augmented_val[source])
            for key, source in COMPARISON_METRICS
        },
        "test": {
            key: float(augmented_test[source])
            for key, source in COMPARISON_METRICS
        },
    }
    return {
        "experiment": "Experiment 4: Controlled Training-Time Data Augmentation",
        "baseline_checkpoint": str(
            BASELINE_CHECKPOINT.relative_to(PROJECT_ROOT)
        ),
        "augmented_checkpoint": str(BEST_PATH.relative_to(PROJECT_ROOT)),
        "comparison_method": (
            "Absolute difference = abs(augmented metric - baseline metric); "
            "no automatic better/worse declaration."
        ),
        "splits": {
            split: {
                metric: {
                    "baseline": baseline[split][metric],
                    "augmented": augmented[split][metric],
                    "absolute_difference": abs(
                        augmented[split][metric] - baseline[split][metric]
                    ),
                }
                for metric, _ in COMPARISON_METRICS
            }
            for split in ("val", "test")
        },
    }


def print_environment(
    config: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    device: torch.device,
) -> None:
    print("EXPERIMENT 4: CONTROLLED TRAINING-TIME DATA AUGMENTATION")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Device: {device}")
    print(f"Seed: {config['seed']}")
    print(f"Dataset sizes: { {split: len(frame) for split, frame in split_frames.items()} }")
    print(f"Feature dimension: {INPUT_DIM}")
    print(f"Number of classes: {NUM_CLASSES}")
    print("Augmentation configuration:")
    print(yaml.safe_dump(config["augmentation"], sort_keys=False).rstrip())
    print("Model configuration:")
    print(
        yaml.safe_dump(
            {
                key: value
                for key, value in config.items()
                if key != "augmentation"
            },
            sort_keys=False,
        ).rstrip()
    )


def smoke_augmentation_config(config: dict[str, Any]) -> dict[str, Any]:
    augmentation = copy.deepcopy(config["augmentation"])
    augmentation["enabled"] = True
    for section in ("temporal_crop", "temporal_scale", "landmark_noise", "frame_dropout"):
        augmentation[section]["enabled"] = True
        augmentation[section]["probability"] = 1.0
    augmentation["temporal_scale"]["min_scale"] = 1.10
    augmentation["temporal_scale"]["max_scale"] = 1.10
    augmentation["landmark_noise"]["std"] = 0.005
    augmentation["frame_dropout"]["frame_probability"] = 0.25
    return augmentation


def run_smoke_test(
    config: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    mapping: dict[str, int],
) -> None:
    """Exercise transforms, forward/backward, and temporary checkpoint I/O."""
    print("Running isolated augmentation smoke test (4 train / 2 val / 2 test).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(config["seed"]))
    augmentation = smoke_augmentation_config(config)
    train_frame = split_frames["train"].head(4)
    val_frame = split_frames["val"].head(2)
    test_frame = split_frames["test"].head(2)
    if min(len(train_frame), len(val_frame), len(test_frame)) == 0:
        raise ValueError("Smoke test needs non-empty train, validation, and test splits.")

    synthetic = np.arange(160 * INPUT_DIM, dtype=np.float32).reshape(160, INPUT_DIM)
    synthetic_out, synthetic_length = apply_temporal_augmentations(
        synthetic, int(config["max_frames"]), augmentation, np.random.default_rng(42)
    )
    apply_landmark_noise_and_dropout(
        synthetic_out, synthetic_length, augmentation, np.random.default_rng(43)
    )
    if synthetic_out.shape != (int(config["max_frames"]), INPUT_DIM):
        raise AssertionError(f"Unexpected augmented shape: {synthetic_out.shape}")
    if not np.isfinite(synthetic_out).all():
        raise AssertionError("Augmentation produced NaN/Inf.")
    if np.array_equal(synthetic_out[:synthetic_length], synthetic[:synthetic_length]):
        raise AssertionError("Forced smoke-test augmentation did not alter the input.")

    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    datasets = {
        "train": LandmarkSequenceDataset(
            train_frame,
            mapping,
            int(config["max_frames"]),
            int(config["seed"]),
            True,
            augmentation,
        ),
        "val": LandmarkSequenceDataset(
            val_frame,
            mapping,
            int(config["max_frames"]),
            int(config["seed"]),
            False,
            config["augmentation"],
        ),
        "test": LandmarkSequenceDataset(
            test_frame,
            mapping,
            int(config["max_frames"]),
            int(config["seed"]),
            False,
            config["augmentation"],
        ),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=(split == "train"),
            num_workers=0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    datasets["train"].set_epoch(1)
    train_sample = datasets["train"][0]
    epoch_one = train_sample["features"]
    datasets["train"].set_epoch(2)
    epoch_two = datasets["train"][0]["features"]
    if torch.equal(epoch_one, epoch_two):
        raise AssertionError("Training augmentation did not vary between epochs.")
    if train_sample["features"].shape != (int(config["max_frames"]), INPUT_DIM):
        raise AssertionError(
            f"Unexpected training tensor shape: {train_sample['features'].shape}"
        )
    if not torch.isfinite(train_sample["features"]).all():
        raise AssertionError("Training tensor contains NaN/Inf.")
    for split in ("val", "test"):
        first = datasets[split][0]["features"]
        second = datasets[split][0]["features"]
        if not torch.equal(first, second):
            raise AssertionError(f"{split} preprocessing is not deterministic.")

    model = build_model(config, device)
    first_parameter = next(model.parameters())
    weights_before = first_parameter.detach().clone()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["label_smoothing"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_metrics = run_epoch(
        model,
        loaders["train"],
        criterion,
        device,
        NUM_CLASSES,
        optimizer,
        scaler,
        float(config["gradient_clip"]),
    )
    if torch.equal(weights_before, first_parameter.detach()):
        raise AssertionError("Smoke training did not update model weights.")
    with torch.no_grad():
        val_metrics = run_epoch(
            model, loaders["val"], criterion, device, NUM_CLASSES
        )
        test_metrics = run_epoch(
            model, loaders["test"], criterion, device, NUM_CLASSES
        )
    if not all(
        np.isfinite(value)
        for metrics in (train_metrics, val_metrics, test_metrics)
        for value in metrics.values()
    ):
        raise AssertionError("Smoke-test metrics contain NaN/Inf.")

    with tempfile.TemporaryDirectory(prefix="isl_augmented_smoke_") as temp_dir:
        checkpoint_path = Path(temp_dir) / "smoke_checkpoint.pt"
        torch.save(
            make_checkpoint(
                model,
                optimizer,
                1,
                val_metrics["macro_f1"],
                mapping,
                config,
            ),
            checkpoint_path,
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "best_val_macro_f1",
            "class_to_index",
            "model_config",
            "augmentation_config",
        }
        if not required.issubset(checkpoint):
            raise AssertionError("Smoke checkpoint is missing required fields.")
        reloaded = build_model(config, torch.device("cpu"))
        reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print(
        "Smoke test passed: augmentation changed inputs across epochs; "
        "tensor shape=(batch, 128, 225); finite forward/loss/backward; "
        "validation/test deterministic; checkpoint saved and reloaded."
    )
    print(
        f"Smoke metrics: train_loss={train_metrics['loss']:.4f}, "
        f"val_loss={val_metrics['loss']:.4f}, test_loss={test_metrics['loss']:.4f}"
    )


def check_results() -> int:
    required_paths = (BEST_PATH, FINAL_PATH, HISTORY_PATH, RESULTS_PATH, COMPARISON_PATH)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Augmented experiment results are incomplete: " + ", ".join(missing)
        )
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    checkpoint = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
    if results["feature_dimension"] != INPUT_DIM:
        raise ValueError("Results feature dimension is not 225.")
    if checkpoint["model_config"]["input_dim"] != INPUT_DIM:
        raise ValueError("Best checkpoint input dimension is not 225.")
    if checkpoint.get("augmentation_config") != results["augmentation_config"]:
        raise ValueError("Checkpoint and results augmentation configs disagree.")
    if not history.get("epochs") or not any(
        "validation" in epoch for epoch in history["epochs"]
    ):
        raise ValueError("Training history contains no validation macro-F1 values.")
    if set(comparison.get("splits", {})) != {"val", "test"}:
        raise ValueError("Comparison report is missing validation/test metrics.")
    print("Augmented experiment outputs are present and internally consistent.")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Best validation macro F1: {results['best_validation_metrics']['macro_f1']:.4f}")
    print(f"Final test macro F1: {results['final_test_metrics']['macro_f1']:.4f}")
    print(f"Best checkpoint: {BEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Results: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return 0
def generate_comparison_only() -> int:
    """Generate Experiment 4 baseline comparison from existing saved outputs."""
    if not BEST_PATH.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {BEST_PATH}")

    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")

    if not HISTORY_PATH.is_file():
        raise FileNotFoundError(f"History file not found: {HISTORY_PATH}")

    baseline = baseline_metrics()

    results = json.loads(
        RESULTS_PATH.read_text(encoding="utf-8")
    )

    augmented_val = results["best_validation_metrics"]
    augmented_test = results["final_test_metrics"]

    comparison = difference_report(
        baseline,
        augmented_val,
        augmented_test,
    )

    COMPARISON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPARISON_PATH.write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("Comparison report generated successfully.")
    print(f"Comparison: {COMPARISON_PATH.relative_to(PROJECT_ROOT)}")

    for split in ("val", "test"):
        print(f"\n{split.upper()} COMPARISON")

        for metric, _ in COMPARISON_METRICS:
            baseline_value = comparison["splits"][split][metric]["baseline"]
            augmented_value = comparison["splits"][split][metric]["augmented"]

            print(
                f"{metric}: "
                f"baseline={baseline_value:.4f} | "
                f"augmented={augmented_value:.4f} | "
                f"absolute_difference="
                f"{comparison['splits'][split][metric]['absolute_difference']:.4f}"
            )

    return 0


def train(config: dict[str, Any], split_frames: dict[str, pd.DataFrame]) -> int:
    existing = [str(path) for path in OUTPUT_PATHS if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing augmented experiment outputs: "
            + ", ".join(existing)
        )
    baseline = baseline_metrics()
    mapping = class_mapping(split_frames["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_environment(config, split_frames, device)
    seed_everything(int(config["seed"]))

    datasets = {
        split: LandmarkSequenceDataset(
            split_frames[split],
            mapping,
            int(config["max_frames"]),
            int(config["seed"]),
            split == "train",
            config["augmentation"],
        )
        for split in SPLITS
    }
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config["seed"]))
    loader_options = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=loader_generator,
            **loader_options,
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **loader_options),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_options),
    }

    model = build_model(config, device)
    print("\nMODEL ARCHITECTURE")
    print(model)
    print(
        "Trainable parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}"
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["label_smoothing"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: dict[str, Any] = {
        "experiment": "Experiment 4: Controlled Training-Time Data Augmentation",
        "seed": int(config["seed"]),
        "device": str(device),
        "dataset_sizes": {split: len(split_frames[split]) for split in SPLITS},
        "feature_dimension": INPUT_DIM,
        "number_of_classes": NUM_CLASSES,
        "model_config": {
            key: value for key, value in config.items() if key != "augmentation"
        }
        | {"input_dim": INPUT_DIM},
        "augmentation_config": copy.deepcopy(config["augmentation"]),
        "epochs": [],
    }
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_validation_metrics: dict[str, float] | None = None
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
            NUM_CLASSES,
            optimizer,
            scaler,
            float(config["gradient_clip"]),
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, loaders["val"], criterion, device, NUM_CLASSES
            )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "learning_rate": learning_rate,
        }
        history["epochs"].append(epoch_record)
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
            f"Val Top-5: {val_metrics['top5']:.4f} | "
            f"Learning Rate: {learning_rate:.7f}"
        )
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_validation_metrics = val_metrics.copy()
            stale_epochs = 0
            torch.save(
                make_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_val_macro_f1,
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

    if best_validation_metrics is None:
        raise RuntimeError("Training finished without a best validation checkpoint.")
    torch.save(
        make_checkpoint(
            model,
            optimizer,
            epoch,
            best_val_macro_f1,
            mapping,
            config,
        ),
        FINAL_PATH,
    )
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history["best_epoch"] = best_epoch
    history["best_validation_metrics"] = best_validation_metrics
    HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    best_checkpoint = torch.load(
        BEST_PATH, map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    with torch.no_grad():
        test_metrics = run_epoch(
            model, loaders["test"], criterion, device, NUM_CLASSES
        )
    result = {
        "experiment": "Experiment 4: Controlled Training-Time Data Augmentation",
        "seed": int(config["seed"]),
        "device": str(device),
        "dataset_sizes": {split: len(split_frames[split]) for split in SPLITS},
        "feature_dimension": INPUT_DIM,
        "number_of_classes": NUM_CLASSES,
        "model_config": history["model_config"],
        "augmentation_config": history["augmentation_config"],
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "final_test_metrics": test_metrics,
        "checkpoint": str(BEST_PATH.relative_to(PROJECT_ROOT)),
    }
    history["final_test_metrics_at_best_validation_checkpoint"] = test_metrics
    HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    comparison = difference_report(
        baseline,
        best_validation_metrics,
        test_metrics,
    )
    COMPARISON_PATH.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print("\nFINAL TEST METRICS (BEST VALIDATION CHECKPOINT)")
    print(
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Accuracy: {test_metrics['accuracy']:.4f} | "
        f"Test Macro F1: {test_metrics['macro_f1']:.4f} | "
        f"Test Weighted F1: {test_metrics['weighted_f1']:.4f} | "
        f"Test Top-3: {test_metrics['top3']:.4f} | "
        f"Test Top-5: {test_metrics['top5']:.4f}"
    )
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_validation_metrics['macro_f1']:.4f}")
    print(f"Best validation accuracy: {best_validation_metrics['accuracy']:.4f}")
    print(f"Best validation top-3: {best_validation_metrics['top3']:.4f}")
    print(f"Best validation top-5: {best_validation_metrics['top5']:.4f}")
    print(f"Best checkpoint: {BEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Final checkpoint: {FINAL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"History: {HISTORY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Results: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Baseline comparison: {COMPARISON_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def main() -> int:
    args = parse_args()

    if args.check_results:
        return check_results()

    if args.generate_comparison:
        return generate_comparison_only()
    config = load_config()
    seed_everything(int(config["seed"]))
    split_frames = load_and_validate_splits()
    mapping = class_mapping(split_frames["train"])
    if args.smoke_test:
        run_smoke_test(config, split_frames, mapping)
        return 0
    return train(config, split_frames)


if __name__ == "__main__":
    raise SystemExit(main())
