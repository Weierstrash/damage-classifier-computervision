"""
predict.py
==========
Run inference on one or more images using the trained damage classifier.

USAGE:
-----
# Predict a single image:
    python predict.py --image path/to/image.png

# Predict with a specific checkpoint:
    python predict.py --image path/to/image.png --checkpoint outputs/best_model.pt

# Run the built-in self-test (no real model needed):
    python predict.py --test

# Predict multiple images and save results to CSV:
    python predict.py --folder data/chips --out results.csv
"""

import argparse
import sys
import time
from pathlib import Path

# ── third-party imports with friendly errors ───────────────────────
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("ERROR: PyTorch not installed.\nRun: pip install torch torchvision")

try:
    from torchvision import transforms
except ImportError:
    sys.exit("ERROR: torchvision not installed.\nRun: pip install torchvision")

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: Pillow not installed.\nRun: pip install Pillow")

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy not installed.\nRun: pip install numpy")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ── label info (thhis matches model.py / dataset.py) ─────────────────

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_CLASSES    = len(DAMAGE_CLASSES)

# Colour for each class (will beused in the visualisation)
CLASS_COLOURS = {
    "no-damage":    "#2ecc71",   # green
    "minor-damage": "#f39c12",   # amber
    "major-damage": "#e67e22",   # orange
    "destroyed":    "#e74c3c",   # red
}


# ── image transform (match dataset.py val transform) ─────────

def get_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])


# ── model loading ──────────────────────────────────────────────────

def load_model(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    """
    Load the trained ResNet18 from a checkpoint saved by train.py.
    Falls back to random weights if no checkpoint is found
    (for testing the pipelinee).
    """
    # Import here so predict.py works even if src/ isn't on sys.path yet
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    try:
        from model import build_model
    except ImportError:
        sys.exit(
            "ERROR: Cannot import src/model.py\n"
            "Make sure predict.py is in the project root (next to src/).\n"
        )

    model = build_model(freeze_backbone=False).to(device)

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_acc = checkpoint.get("val_acc", "unknown")
        epoch   = checkpoint.get("epoch",   "unknown")
        print(f"  Loaded checkpoint: {checkpoint_path.name}")
        print(f"  Trained for {epoch} epochs | val_acc = {val_acc:.3f}" if isinstance(val_acc, float) else f"  Epoch: {epoch}")
    else:
        print(f"  WARNING: No checkpoint found at {checkpoint_path}")
        print(f"  Using random weights (run train.py first for real predictions).")

    model.eval()
    return model


# ── core prediction function ───────────────────────────────────────

def predict_image(
    image_path:  str | Path,
    model:       torch.nn.Module,
    device:      torch.device,
    image_size:  int = 224,
) -> dict:
    """
    Run inference on a single image.

    Returns a dict with:
        label       (str)   predicted class name
        label_id    (int)   predicted class index 0-3
        confidence  (float) probability of predicted class 0.0-1.0
        probs       (list)  probability for each of the 4 classes
        image_path  (str)
        time_ms     (float) inference time in milliseconds
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and preprocess
    image     = Image.open(image_path).convert("RGB")
    transform = get_transform(image_size)
    tensor    = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)                       # (1, 4)
        probs  = F.softmax(logits, dim=1)[0]         # (4,)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    label_id   = probs.argmax().item()
    label      = DAMAGE_CLASSES[label_id]
    confidence = probs[label_id].item()

    return {
        "label":      label,
        "label_id":   label_id,
        "confidence": confidence,
        "probs":      [round(p.item(), 4) for p in probs],
        "image_path": str(image_path),
        "time_ms":    round(elapsed_ms, 1),
    }


# ── visualisation ──────────────────────────────────────────────────

def visualise_prediction(result: dict, save_path: str | Path | None = None) -> None:
    """
    Show (and optionally save) a figure with the image and
    a horizontal probability bar chart side by side.
    """
    if not HAS_MATPLOTLIB:
        print("  (install matplotlib to see visualisation: pip install matplotlib)")
        return

    img   = Image.open(result["image_path"]).convert("RGB")
    probs = result["probs"]
    label = result["label"]
    conf  = result["confidence"]
    colour = CLASS_COLOURS[label]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left — the image
    ax1.imshow(np.array(img))
    ax1.set_title(
        f"Predicted: {label}\nConfidence: {conf:.1%}",
        fontsize=12, fontweight="bold", color=colour,
    )
    ax1.axis("off")

    # Right — probability bars
    bar_colours = [CLASS_COLOURS[c] for c in DAMAGE_CLASSES]
    bars = ax2.barh(DAMAGE_CLASSES, probs, color=bar_colours, edgecolor="white")

    # Add percentage labels inside bars
    for bar, p in zip(bars, probs):
        ax2.text(
            max(p - 0.03, 0.01), bar.get_y() + bar.get_height() / 2,
            f"{p:.1%}", va="center", ha="right", fontsize=10,
            color="white", fontweight="bold",
        )

    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Probability")
    ax2.set_title("Class probabilities", fontsize=12)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        Path(result["image_path"]).name,
        fontsize=10, color="gray", y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Visualisation saved → {save_path}")

    plt.show()
    plt.close(fig)


# ── batch prediction over a folder ────────────────────────────────

def predict_folder(
    folder:          str | Path,
    model:           torch.nn.Module,
    device:          torch.device,
    out_csv:         str | Path | None = None,
    image_size:      int = 224,
    extensions:      tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> list[dict]:
    """
    Run inference on every image in a folder. Returns a list of result dicts.
    Optionally saves results to a CSV.
    """
    folder = Path(folder)
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in extensions])

    if not images:
        print(f"  No images found in {folder}")
        return []

    print(f"  Running inference on {len(images)} images...")
    results = []
    for i, img_path in enumerate(images, 1):
        try:
            result = predict_image(img_path, model, device, image_size)
            results.append(result)
            print(f"  [{i:>4}/{len(images)}] {img_path.name:<50} "
                  f"{result['label']:<16} ({result['confidence']:.1%})")
        except Exception as e:
            print(f"  [{i:>4}/{len(images)}] SKIP {img_path.name}: {e}")

    if out_csv and HAS_PANDAS and results:
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\n  Results saved → {out_csv}")

    return results


# ── self-test (no real model / data needed) ───────────────────────

def run_self_test() -> None:
    """
    Creates a random 224x224 RGB image, runs it through the model
    (random weights), and prints + visualises the result.
    Lets you verify the whole pipeline works before you have real data.
    """
    import tempfile

    print("\n" + "="*55)
    print("  Self-test mode (random weights + synthetic image)")
    print("="*55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Build model with random weights (no checkpoint needed)
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    try:
        from model import build_model
    except ImportError:
        sys.exit("ERROR: Cannot import src/model.py — is predict.py in the project root?")

    model = build_model(freeze_backbone=False).to(device)
    model.eval()
    print("  Model: ResNet18 (random weights — run train.py for real predictions)")

    # Create a synthetic satellite-style image (random noise + green-ish tint)
    np.random.seed(42)
    arr = np.random.randint(60, 180, (224, 224, 3), dtype=np.uint8)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + 30, 0, 255)  # slight green bias

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.fromarray(arr).save(tmp.name)
        test_image_path = tmp.name

    print(f"\n  Test image: {test_image_path}")

    result = predict_image(test_image_path, model, device)

    print("\n  ── Result ──────────────────────────────────")
    print(f"  Predicted class : {result['label']}")
    print(f"  Confidence      : {result['confidence']:.1%}")
    print(f"  Inference time  : {result['time_ms']} ms")
    print("\n  Class probabilities:")
    for cls, prob in zip(DAMAGE_CLASSES, result["probs"]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<18} {prob:.3f}  {bar}")
    print("  ────────────────────────────────────────────")

    if HAS_MATPLOTLIB:
        save_path = Path("outputs/test_prediction.png")
        save_path.parent.mkdir(exist_ok=True)
        visualise_prediction(result, save_path=save_path)
    else:
        print("\n  (install matplotlib for visualisation: pip install matplotlib)")

    print("\n  Self-test passed! Pipeline is working correctly.")
    print("  Next step: run train.py, then predict with --image\n")


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict building damage class from satellite image chips."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to a single image file.")
    group.add_argument("--folder", help="Path to a folder of images (batch mode).")
    group.add_argument("--test",   action="store_true",
                       help="Run built-in self-test (no checkpoint needed).")

    parser.add_argument("--checkpoint", default="outputs/best_model.pt",
                        help="Path to model checkpoint (default: outputs/best_model.pt).")
    parser.add_argument("--out",  default=None,
                        help="(batch mode) Save results to this CSV path.")
    parser.add_argument("--save_plot", default=None,
                        help="(single image) Save visualisation PNG to this path.")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size fed to the model (default: 224).")
    args = parser.parse_args()

    # ── self-test ──
    if args.test:
        run_self_test()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print(f"\nLoading model...")
    model = load_model(args.checkpoint, device)

    # ── single image ──
    if args.image:
        print(f"\nRunning inference on: {args.image}\n")
        result = predict_image(args.image, model, device, args.image_size)

        print("── Result ──────────────────────────────────────")
        print(f"  Image      : {Path(result['image_path']).name}")
        print(f"  Prediction : {result['label']}")
        print(f"  Confidence : {result['confidence']:.1%}")
        print(f"  Time       : {result['time_ms']} ms")
        print("\n  Class probabilities:")
        for cls, prob in zip(DAMAGE_CLASSES, result["probs"]):
            bar = "█" * int(prob * 30)
            print(f"    {cls:<18} {prob:.3f}  {bar}")
        print("────────────────────────────────────────────────")

        save_path = args.save_plot or "outputs/prediction.png"
        Path(save_path).parent.mkdir(exist_ok=True)
        visualise_prediction(result, save_path=save_path)

    # ── batch folder ──
    elif args.folder:
        print(f"\nBatch prediction on folder: {args.folder}\n")
        predict_folder(
            folder     = args.folder,
            model      = model,
            device     = device,
            out_csv    = args.out,
            image_size = args.image_size,
        )


if __name__ == "__main__":
    main()