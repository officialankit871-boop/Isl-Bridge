import os
import numpy as np
import pandas as pd

# =========================
# LOAD DATA
# =========================

manifest = pd.read_csv("data/manifests/manifest.csv")
train = pd.read_csv("data/splits/train.csv")
val = pd.read_csv("data/splits/val.csv")
test = pd.read_csv("data/splits/test.csv")

splits = pd.concat([train, val, test], ignore_index=True)

# =========================
# DATASET CONSISTENCY
# =========================

print("=== DATASET CONSISTENCY ===")

print("Manifest videos:", len(manifest))
print("Manifest classes:", manifest["label_id"].nunique())

print("Split videos:", len(splits))
print("Split classes:", splits["label_id"].nunique())

print("Train / Val / Test:", len(train), len(val), len(test))

print(
    "Duplicate split video paths:",
    splits["video_path"].duplicated().sum()
)

print(
    "Manifest missing labels:",
    manifest["label_id"].isna().sum()
)

print(
    "Split missing labels:",
    splits["label_id"].isna().sum()
)

print(
    "Labels in manifest but not splits:",
    set(manifest["label_id"]) - set(splits["label_id"])
)

print(
    "Labels in splits but not manifest:",
    set(splits["label_id"]) - set(manifest["label_id"])
)

# =========================
# LANDMARK CHECK
# =========================

print("\n=== LANDMARKS ===")

landmark_dir = "data/processed/landmarks"

files = [
    f for f in os.listdir(landmark_dir)
    if f.endswith(".npy")
]

print("Landmark files:", len(files))

invalid = 0
shapes = set()

for filename in files:

    path = os.path.join(landmark_dir, filename)

    try:
        arr = np.load(path)

        if arr.ndim == 2:
            shapes.add(arr.shape[1])
        else:
            shapes.add(-1)

        if (
            arr.ndim != 2
            or arr.shape[1] != 225
            or not np.isfinite(arr).all()
        ):
            invalid += 1

    except Exception:
        invalid += 1

print("Unique feature dimensions:", shapes)
print("Invalid landmark files:", invalid)

# =========================
# FINAL RESULT
# =========================

print("\n=== FINAL RESULT ===")

if (
    len(manifest) == 687
    and manifest["label_id"].nunique() == 101
    and len(splits) == 687
    and splits["label_id"].nunique() == 101
    and splits["video_path"].duplicated().sum() == 0
    and manifest["label_id"].isna().sum() == 0
    and splits["label_id"].isna().sum() == 0
    and set(manifest["label_id"]) == set(splits["label_id"])
    and len(files) == 687
    and invalid == 0
    and shapes == {225}
):
    print("PASS: Dataset pipeline is consistent.")
else:
    print("CHECK REQUIRED: One or more consistency checks failed.")