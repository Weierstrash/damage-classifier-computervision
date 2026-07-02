"""
dataset.py — PyyTorch Dataset and DataLoader helpers for the xBD
building damage classification project.

Expected folder layout
----------------------
data/
  labels.csv  v        # columns: filename, label  (label = 0..3)
  chips/
    <image_name>.png  # post-disaster 512x512 RGB chips

labels.csv example
------------------
filename,label
hurricane-michael_00000001_post.png,0
hurricane-michael_00000002_post.png,2
...

Label mapping (matches model.py DAMAGE_CLASSES):
    0  no-damage
    1  minor-damage
    2  major-damage
    3  destroyed
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# ------------------------------------------------------------------
# Image transforms
# ------------------------------------------------------------------

# ImageNet mean/std — correct for ResNet18 pretrained weights
# This will be used to normalize the dataset!
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Augmentation pipeline for training.
    Horizontal + vertical flips, colour jitter, and random rotation
    are all label-preserving for satellite imagery.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic pipeline for validation and inference.
    No augmentation — only resize, centre-crop, and normalise.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation for visualisation.
    Input:  (C, H, W) float tensor
    Output: (C, H, W) float tensor clamped to [0, 1]
    """
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class DamageDataset(Dataset):
    """
    Loads (image, label) pairs from a CSV + an image directory.

    Args:
        dataframe:   A pandas DataFrame with columns 'filename' and 'label'.
        image_dir:   Directory that contains the image files.
        transform:   torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.df        = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row      = self.df.iloc[idx]
        img_path = self.image_dir / row["filename"]

        # Open as RGB — handles grayscale and RGBA chips transparently
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])
        return image, label


# ------------------------------------------------------------------
# Train / val split helper
# ------------------------------------------------------------------

def make_splits(
    labels_csv: str | Path,
    val_size: float = 0.2, # standard
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read labels.csv and return (train_df, val_df).

    Uses stratified splitting so each damage class is proportionally
    represented in both splits — important for imbalanced xBD data.
    """
    df = pd.read_csv(labels_csv)

    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError("labels.csv must have columns: 'filename', 'label'")

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ------------------------------------------------------------------
# Weighted sampler (handles class imbalance)
# ------------------------------------------------------------------

def make_weighted_sampler(dataset: DamageDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so that each mini-batch sees
    roughly equal numbers of all four damage classes.

    xBD is heavily skewed toward 'no-damage', so this prevents the
    model from just predicting the majority class.
    """
    labels  = dataset.df["label"].values
    classes, counts = np.unique(labels, return_counts=True)

    # Weight per class = 1 / count; weight per sample = its class weight
    class_weights  = 1.0 / counts.astype(float)
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float),
        num_samples = len(sample_weights),
        replacement = True,
    )


# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------

def build_dataloaders(
    labels_csv: str | Path,
    image_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    One-call factory that returns (train_loader, val_loader).

    The training loader uses a WeightedRandomSampler to balance classes.
    The validation loader uses a plain sequential sampler.

    Args:
        labels_csv:   Path to labels.csv.
        image_dir:    Directory containing the chip images.
        batch_size:   Number of samples per mini-batch.
        image_size:   Spatial size (H = W) passed to the transforms.
        num_workers:  Dataloader worker processes (set 0 on Windows).
        val_size:     Fraction of data held out for validation.
        random_state: Seed for reproducible splits.

    Returns:
        (train_loader, val_loader)
    """
    train_df, val_df = make_splits(labels_csv, val_size, random_state)

    train_dataset = DamageDataset(train_df, image_dir, get_train_transforms(image_size))
    val_dataset   = DamageDataset(val_df,   image_dir, get_val_transforms(image_size))

    sampler = make_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,       # replaces shuffle=True
        num_workers = num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    print(f"Class distribution (train):\n{train_df['label'].value_counts().sort_index()}")

    return train_loader, val_loader


# ------------------------------------------------------------------
# Ssanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, shutil

    # Create a tiny fake dataset to verify the pipeline runs end-to-end
    # Starting with a temporary directory
    tmp = Path(tempfile.mkdtemp())
    chips_dir = tmp / "chips"
    chips_dir.mkdir()

    rows = []
    for i in range(40):
        label = i % 4
        fname = f"chip_{i:04d}.png"
        # Save a random 64x64 RGB image
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3),dtype = np.uint8)
        )
        img.save(chips_dir / fname)
        rows.append({"filename": fname, "label": label})
       

    csv_path = tmp / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    train_loader, val_loader = build_dataloaders(
        labels_csv  = csv_path,
        image_dir   = chips_dir,
        batch_size  = 8,
        num_workers = 0,   # 0 for inline testing
    )


    images, labels = next(iter(train_loader))
    print(f"\nBatch image shape: {images.shape}")   # (8, 3, 224, 224)
    print(f"Batch label shape: {labels.shape}")     # (8,)
    print(f"Label values:      {labels.tolist()}")

    shutil.rmtree(tmp)
    print("\nAll checks passed.")