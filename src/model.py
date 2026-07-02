"""
model.py — ResNet18 fine-tuned for 4-class building damage classification.

Classes:
    0: no-damage
    1: minor-damage
    2: major-damage
    3: destroyed
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# ------------------------------------------------------------------
# Class labels (shared across model, training, and inference)
# ------------------------------------------------------------------

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_CLASSES = len(DAMAGE_CLASSES) 


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def build_model(freeze_backbone: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    Load a pretrained ResNet18 and replace the final fully-connected
    layer with a 4-class classification head.

    Args:
        freeze_backbone: If True, freeze all layers except the new head.
                         Set to False for full fine-tuning (slower, needs
                         more data or a lower LR). By default, this is set to
                         True.
        dropout:         Dropout probability before the linear layer.

    Returns:
        A torch.nn.Module ready for training or inference.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final FC layer
    # ResNet18 fc: Linear(512 -> 1000)  →  we replace with our head (4 classifier)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )

    return model


def load_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Rebuild the model architecture and load saved weights.

    Args:
        checkpoint_path: Path to the .pt file saved by train.py.
        device:          torch.device to map the weights onto.

    Returns:
        Model in eval mode with weights loaded.
    """
    model = build_model(freeze_backbone=False)  # all layers unfrozen for inference
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(freeze_backbone=True).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}  ({100*trainable/total:.1f}%)")

    # Forward pass with a dummy batch
    dummy = torch.randn(4, 3, 224, 224).to(device)
    logits = model(dummy)
    print(f"Output shape: {logits.shape}")  # expected: (4, 4)