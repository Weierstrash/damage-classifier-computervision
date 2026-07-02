# Building Damage Classifier

A computer vision model that classifies building damage severity from post-disaster satellite imagery. Fine-tunes a pretrained ResNet18 on the xBD datase to predict one of four damage classes per building chip.

## Classes

| Label | Class | Description |
|-------|-------|-------------|
| 0 | `no-damage` | Building intact |
| 1 | `minor-damage` | Partial roof or wall damage |
| 2 | `major-damage` | Significant structural damage |
| 3 | `destroyed` | Building collapsed or gone |

## Results

| Metric | Value |
|--------|-------|
| Val accuracy | ~70% (500 chips, 15 epochs) |
| Model | ResNet18 (pretrained ImageNet) |
| Training time | ~45 min on CPU |


## Quickstart

**1. Install dependencies**
```bash
pipenv install -r requirements.txt
```

**2. Get the data**

Register at (https://xview2.org) and download the `train` split. Then:

```bash
python prepare_data.py --xbd_dir /path/to/train --max_chips 500
```

This reads the xBD JSON labels, crops one 224×224 chip per building, and writes `data/chips/` and `data/labels.csv`.

**3. Train**
```bash
python src/train.py
```

Saves `outputs/best_model.pt`, `outputs/training_curves.png`, and `outputs/confusion_matrix.png`.

**4. Predict**
```bash
# Single image
python predict.py --image data/chips/chip_000001.png

# Whole folder → CSV
python predict.py --folder data/chips --out results.csv

# Self-test (no checkpoint needed)
python predict.py --test
```

## HowIt Workss

1. **Data** — xBD provides 1024×1024 post-disaster satellite images with polygon annotations per building. `prepare_data.py` reads these JSON files, finds each building's bounding box, and crops a 224×224 chip centred on it.

2. **Model** — ResNet18 pretrained on ImageNet. The final fully-connected layer is replaced with `Dropout → Linear(512 → 4)`. The backbone is frozen for the first epochs, then unfrozen at a lower learning rate for full fine-tuning.

3. **Class imbalance** — xBD is heavily skewed toward `no-damage`. Training uses a `WeightedRandomSampler` to balance mini-batches across all four classes.

4. **Output** — `predict.py` returns the predicted class, confidence score, and a bar chart of all four class probabilities.


