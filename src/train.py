"""
train.py — Training loop for the building damage classifier.

Usage
-----
    python src/train.py

Outputs (all written to ./outputs/)
------------------------------------
    best_model.pt          Checkpoint with best val accuracy
    training_curves.png    Loss + accuracy plot
    confusion_matrix.png   Per-class confusion matrix on val set
    results.txt            Final metrics summary
"""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset import build_dataloaders
from model import DAMAGE_CLASSES, NUM_CLASSES, build_model


# ------------------------------------------------------------------
# Config — edit these to match your setup
# ------------------------------------------------------------------

CONFIG = {
    "labels_csv":    "data/labels.csv",
    "image_dir":     "data/chips",
    "output_dir":    "outputs",
    "batch_size":    32,
    "image_size":    224,
    "num_workers":   4,         
    "epochs":        15,
    "lr":            1e-3,      # head LR (backbone frozen)
    "unfreeze_epoch": 8,        # unfreeze backbone at this epoch
    "backbone_lr":   1e-5,      # backbone LR after unfreezing
    "weight_decay":  1e-4,
    "val_size":      0.2,
    "random_state":  42,
    "patience":      5,         # early stopping patience
}


# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run validation. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    running_loss = 0.0
    correct = total = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        running_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def plot_training_curves(
    history: dict,
    output_path: str | Path,
) -> None:
    """Save a 2-panel figure: loss (left) and accuracy (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved training curves → {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save aa norrmalised confusion matrix as a PNG."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=DAMAGE_CLASSES)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title("Normalised confusion matrix (val set)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix  → {output_path}")


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader = build_dataloaders(
        labels_csv  = CONFIG["labels_csv"],
        image_dir   = CONFIG["image_dir"],
        batch_size  = CONFIG["batch_size"],
        image_size  = CONFIG["image_size"],
        num_workers = CONFIG["num_workers"],
        val_size    = CONFIG["val_size"],
        random_state= CONFIG["random_state"],
    )

    # Model — backbone frozen, only head trains initially
    model = build_model(freeze_backbone=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = CONFIG["lr"],
        weight_decay = CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"]
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc   = 0.0
    patience_count = 0
    best_preds = best_labels = None

    print(f"\nTraining for up to {CONFIG['epochs']} epochs "
          f"(early stop patience={CONFIG['patience']})\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        # Unfreeze backbone mid-training for full fine-tuning
        if epoch == CONFIG["unfreeze_epoch"]:
            print(f"\n[Epoch {epoch}] Unfreezing backbone — switching to lr={CONFIG['backbone_lr']}")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr           = CONFIG["backbone_lr"],
                weight_decay = CONFIG["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = CONFIG["epochs"] - epoch + 1,
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{CONFIG['epochs']}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"({elapsed:.1f}s)"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds   = val_preds
            best_labels  = val_labels
            patience_count = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                    "config":           CONFIG,
                },
                out_dir / "best_model.pt",
            )
            print(f"  ✓ New best val_acc={val_acc:.3f} — checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {CONFIG['patience']} epochs)")
                break

    # ------------------------------------------------------------------
    # Postt-training outputs
    # ------------------------------------------------------------------
    print("\n--- Final results ---")

    plot_training_curves(history, out_dir / "training_curves.png")
    plot_confusion_matrix(best_labels, best_preds, out_dir / "confusion_matrix.png")

    report = classification_report(
        best_labels, best_preds, target_names=DAMAGE_CLASSES, digits=3
    )
    print(report)

    results_path = out_dir / "results.txt"
    with open(results_path, "w") as f:
        f.write(f"Best val accuracy: {best_val_acc:.4f}\n\n")
        f.write(report)
    print(f"Saved results summary → {results_path}")
    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    train()