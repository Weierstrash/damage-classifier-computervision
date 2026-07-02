"""
prepare_data.py
===============
Reads raw xBD data (images + JSON labels), crops one 224x224 chip
per building, and writes:

    data/chips/      <-- one PNG per building
    data/labels.csv  <-- filename, label (0-3)


HOW TO RUN
----------
Point --xbd_dir at whatever folder contains your downloaded xBD data.
The script auto-detects the layout — works for Kaggle, xview2.org,
and per-disaster layouts.

    python prepare_data.py --xbd_dir /path/to/your/xbd/download

Optional flags:
    --max_chips 500     how many chips to generate (default: 500)
    --chip_size 224     output image size in pixels (default: 224)
    --out_dir   data    where to write chips/ and labels.csv

LABEL MAPPING
-------------
    0  no-damage
    1  minor-damage
    2  major-damage
    3  destroyed
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# ── third-party (install if missing) ──────────────────────────────
try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy not installed. Run: pip install numpy")

try:
    import pandas as pd
except ImportError:
    sys.exit("ERROR: pandas not installed. Run: pip install pandas")

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: Pillow not installed. Run: pip install Pillow")

try:
    from shapely import wkt as shapely_wkt
except ImportError:
    sys.exit("ERROR: shapely not installed. Run: pip install shapely")


# ── constants ──────────────────────────────────────────────────────

LABEL_MAP = {
    "no-damage":    0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed":    3,
}
SKIP_LABELS = {"un-classified", "background", "unclassified"}


# ── step 1: find images/ and labels/ folders ──────────────────────

def find_image_and_label_dirs(root: Path) -> list[tuple[Path, Path]]:
    """
    Search the download folder for pairs of images/ + labels/ directories.
    Returns a list of (image_dir, label_dir) pairs.

    Handles all known xBD layouts:
      Layout A (Kaggle / xview2.org):
        train/
          images/  labels/

      Layout B (per-disaster, GitHub baseline):
        hurricane-michael/
          images/  labels/
        palu-tsunami/
          images/  labels/
    """
    pairs = []

    # Walk every subdirectory looking for a "labels" folder
    for label_dir in sorted(root.rglob("labels")):
        if not label_dir.is_dir():
            continue
        # Check for a sibling "images" folder
        image_dir = label_dir.parent / "images"
        if image_dir.is_dir():
            pairs.append((image_dir, label_dir))

    return pairs


def explain_what_was_found(root: Path, pairs: list) -> None:
    """Print a clear summary of what the script found."""
    print(f"\nSearched: {root}")
    if not pairs:
        print("  (no images/ + labels/ pairs found)")
    else:
        print(f"  Found {len(pairs)} images/+labels/ pair(s):")
        for img_dir, lbl_dir in pairs:
            n_imgs  = len(list(img_dir.glob("*.png")))
            n_jsons = len(list(lbl_dir.glob("*.json")))
            print(f"    {lbl_dir.parent.name}/")
            print(f"      images/ — {n_imgs} PNGs")
            print(f"      labels/ — {n_jsons} JSONs")


# ── step 2: parse one xBD JSON label file ─────────────────────────

def parse_json(json_path: Path) -> list[dict]:
    """
    Read one xBD JSON file and return a list of building records.

    xBD JSON structure:
    {
      "features": {
        "xy": [
          {
            "properties": { "subtype": "no-damage" },
            "wkt": "POLYGON ((x1 y1, x2 y2, ...))"
          },
          ...
        ]
      }
    }
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [skip] could not read {json_path.name}: {e}")
        return []

    features = data.get("features", {}).get("xy", [])
    records  = []

    for feat in features:
        props   = feat.get("properties", {})
        subtype = props.get("subtype", "un-classified").strip().lower()

        # Skip unlabelled / background polygons
        if subtype in SKIP_LABELS or subtype not in LABEL_MAP:
            continue

        wkt_str = feat.get("wkt", "")
        if not wkt_str:
            continue

        try:
            polygon = shapely_wkt.loads(wkt_str)
        except Exception:
            continue

        if polygon.is_empty:
            continue

        records.append({
            "polygon": polygon,
            "label":   LABEL_MAP[subtype],
        })

    return records


# ── step 3: crop a chip around one building polygon ───────────────

def crop_chip(
    image: Image.Image,
    polygon,
    chip_size: int,
    pad_factor: float = 0.4,
) -> Image.Image | None:
    """
    Crop a square chip centred on a building polygon.
    Returns None if the building is too small or out of bounds.
    """
    minx, miny, maxx, maxy = polygon.bounds
    bw = maxx - minx
    bh = maxy - miny

    if bw < 4 or bh < 4:      # skip pixel-sized artefacts
        return None

    pad  = max(bw, bh) * pad_factor
    cx   = (minx + maxx) / 2
    cy   = (miny + maxy) / 2
    half = max(bw, bh) / 2 + pad

    img_w, img_h = image.size
    left  = max(0, int(cx - half))
    upper = max(0, int(cy - half))
    right = min(img_w, int(cx + half))
    lower = min(img_h, int(cy + half))

    if right <= left or lower <= upper:
        return None

    chip = image.crop((left, upper, right, lower))
    chip = chip.resize((chip_size, chip_size), Image.LANCZOS)
    return chip


# ── step 4: main processing loop ──────────────────────────────────

def process(
    pairs:      list[tuple[Path, Path]],
    out_chips:  Path,
    chip_size:  int,
    max_chips:  int,
    seed:       int,
) -> pd.DataFrame:

    out_chips.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # Collect all post-disaster JSON files across all pairs
    all_jsons = []
    for image_dir, label_dir in pairs:
        jsons = list(label_dir.glob("*post_disaster*.json"))
        if not jsons:
            # Some layouts use "post" without underscore
            jsons = list(label_dir.glob("*post*.json"))
        if not jsons:
            # Fall back to ALL json files in this label dir
            jsons = list(label_dir.glob("*.json"))
        for j in jsons:
            all_jsons.append((j, image_dir))

    if not all_jsons:
        sys.exit(
            "\nERROR: No JSON label files found.\n"
            "Make sure your download includes a labels/ folder with .json files.\n"
        )

    print(f"\nFound {len(all_jsons)} JSON label files total.")
    print(f"Generating up to {max_chips} chips...\n")

    random.shuffle(all_jsons)   # spread across disaster types

    rows          = []
    chips_written = 0

    for json_path, image_dir in all_jsons:
        if chips_written >= max_chips:
            break

        records = parse_json(json_path)
        if not records:
            continue

        # Find the matching image — same stem, .png extension
        img_path = image_dir / (json_path.stem + ".png")
        if not img_path.exists():
            # Some layouts store images without the _disaster suffix
            alt = image_dir / (json_path.stem.replace("_post_disaster", "") + ".png")
            if alt.exists():
                img_path = alt
            else:
                continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [skip] cannot open {img_path.name}: {e}")
            continue

        random.shuffle(records)

        for rec in records:
            if chips_written >= max_chips:
                break

            chip = crop_chip(image, rec["polygon"], chip_size)
            if chip is None:
                continue

            chip_name = f"chip_{chips_written:06d}.png"
            chip.save(out_chips / chip_name)
            rows.append({"filename": chip_name, "label": rec["label"]})
            chips_written += 1

        if chips_written > 0 and chips_written % 100 == 0:
            print(f"  {chips_written}/{max_chips} chips written...")

    return pd.DataFrame(rows)


# ── step 5: print class distribution ──────────────────────────────

def print_distribution(df: pd.DataFrame) -> None:
    label_names = {v: k for k, v in LABEL_MAP.items()}
    total = len(df)
    print("\nClass distribution in labels.csv:")
    print(f"  {'Class':<18} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*36}")
    for i, name in sorted(label_names.items()):
        count = (df["label"] == i).sum()
        pct   = 100 * count / total if total else 0
        print(f"  {name:<18} {count:>6}  {pct:>5.1f}%")
    print(f"  {'-'*36}")
    print(f"  {'TOTAL':<18} {total:>6}  100.0%")


# ── entry point ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build chips/ and labels.csv from raw xBD data."
    )
    parser.add_argument(
        "--xbd_dir",
        required=True,
        help="Path to your downloaded xBD folder (any layout).",
    )
    parser.add_argument(
        "--max_chips", type=int, default=500,
        help="Max chips to generate (default: 500).",
    )
    parser.add_argument(
        "--chip_size", type=int, default=224,
        help="Output chip size in pixels (default: 224).",
    )
    parser.add_argument(
        "--out_dir", default="data",
        help="Where to write chips/ and labels.csv (default: data/).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    xbd_dir   = Path(args.xbd_dir).expanduser().resolve()
    out_dir   = Path(args.out_dir).expanduser().resolve()
    out_chips = out_dir / "chips"
    out_csv   = out_dir / "labels.csv"

    if not xbd_dir.exists():
        sys.exit(f"\nERROR: folder not found: {xbd_dir}\n")

    print("=" * 60)
    print("  xBD data preparation")
    print("=" * 60)

    # Auto-detect layout
    pairs = find_image_and_label_dirs(xbd_dir)
    explain_what_was_found(xbd_dir, pairs)

    if not pairs:
        print(
            "\nERROR: Could not find any images/ + labels/ folder pairs.\n"
            "\nYour folder structure should look like ONE of these:\n"
            "\n  Layout A (Kaggle):\n"
            "    your_download/\n"
            "      train/\n"
            "        images/   <-- PNG files\n"
            "        labels/   <-- JSON files\n"
            "\n  Layout B (per-disaster):\n"
            "    your_download/\n"
            "      hurricane-michael/\n"
            "        images/\n"
            "        labels/\n"
            "      palu-tsunami/\n"
            "        images/\n"
            "        labels/\n"
            "\nRun this to show what's inside your download folder:\n"
            f"  find {xbd_dir} -type d | head -30\n"
        )
        sys.exit(1)

    # Generate chips
    df = process(
        pairs     = pairs,
        out_chips = out_chips,
        chip_size = args.chip_size,
        max_chips = args.max_chips,
        seed      = args.seed,
    )

    if df.empty:
        sys.exit(
            "\nERROR: 0 chips were generated.\n"
            "The JSON files were found but no valid building polygons "
            "couldbe matched to images.\n"
            "Check that your images/ and labels/ folders contain matching filenames.\n"
        )

    # Write labels.csv
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")
    print(f"  Chips written : {len(df)}")
    print(f"  Chips folder  : {out_chips}")
    print(f"  labels.csv    : {out_csv}")
    print_distribution(df)
    print(f"\nNext step:")
    print(f"  python src/train.py\n")


if __name__ == "__main__":
    main()