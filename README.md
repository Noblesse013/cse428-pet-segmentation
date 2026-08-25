# CSE428 — Multi-Task Pet Segmentation & Breed Classification

Multi-task deep-learning project using the **Oxford-IIIT Pet dataset**.
The model jointly learns **binary pet segmentation** (foreground vs. background)
and **breed classification** (37 classes, Cat/Dog).

Two architectures are implemented and compared:

1. **Base U-Net** + classification head
2. **Attention U-Net** + classification head

Both share a common encoder and train segmentation + classification together.

---

## Requirements

- **NVIDIA CUDA-enabled GPU** and a CUDA-enabled PyTorch installation are **required**.
  The code will raise an error if no CUDA GPU is detected.
- Python 3.9+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The Oxford-IIIT Pet dataset must already be present in the repository:

```
images/           # JPG pet images
annotations/      # Annotation files (list.txt, trainval.txt, test.txt, trimaps/)
```

Do **not** place the dataset download zip/tar here — extract `images/` and
`annotations/` directly into the project root.

---

## Project Structure

```
project_root/
├── images/                  # Pet images (JPG)
├── annotations/             # Annotation metadata + trimap PNGs
│   └── annotations/
│       ├── list.txt         # Full image list with class/species IDs
│       ├── trainval.txt     # Official trainval split
│       ├── test.txt         # Official test split
│       └── trimaps/         # Per-pixel trimap annotations
│
├── src/                     # Source modules
│   ├── dataset.py           # Dataset class + split logic
│   ├── transforms.py        # Augmentation + preprocessing
│   ├── losses.py            # DiceLoss + multi-task loss
│   ├── metrics.py           # IoU, Dice, Precision, Recall, F1
│   ├── train_utils.py       # Training & validation loops
│   ├── evaluation.py        # Full evaluation + table printing
│   ├── visualization.py     # Plots + demo overlay
│   └── models/
│       ├── blocks.py        # DoubleConv, DownBlock, UpBlock
│       ├── unet.py          # Base U-Net + classifier
│       ├── attention_blocks.py  # AttentionGate
│       └── attention_unet.py # Attention U-Net + classifier
│
├── checkpoints/             # Saved model weights (created during training)
├── results/                 # CSVs, plots, demo images
│
├── config.py                # Central configuration
├── explore_dataset.py       # Dataset exploration (3x3 grid)
├── train_unet.py            # Train Base U-Net
├── train_attention_unet.py  # Train Attention U-Net
├── evaluate.py              # Evaluate trained checkpoints
├── demo.py                  # Visualise single-image predictions
├── requirements.txt
└── README.md
```

---

## Usage

### 1. Explore the dataset

```bash
python explore_dataset.py
```

Prints dataset statistics and saves a 3x3 random sample grid to
`results/exploration_samples.png`.

### 2. Train Base U-Net

```bash
python train_unet.py
```

Saves:
- `checkpoints/best_unet.pth`
- `results/unet_history.csv`
- `results/unet_*.png` (training curves)

### 3. Train Attention U-Net

```bash
python train_attention_unet.py
```

Saves:
- `checkpoints/best_attention_unet.pth`
- `results/attention_unet_history.csv`
- `results/attention_unet_*.png`

### 4. Evaluate a trained model

```bash
python evaluate.py --model unet
python evaluate.py --model attention_unet
```

Prints a formatted table with IoU, Dice, Pixel Accuracy, Accuracy,
Precision, Recall, and F1 for train / validation / test splits.

### 5. Demo: visualise predictions

```bash
python demo.py --model unet --index 145
python demo.py --model attention_unet --index 145
```

Saves an image showing the original, true mask overlay, and predicted
mask overlay with breed and IoU information.

---

## Architecture

### Base U-Net

- 4-level encoder (DoubleConv → MaxPool)
- Bottleneck (DoubleConv → MaxPool)
- 4-level decoder (Upsample → Skip Concat → DoubleConv)
- Segmentation head: 1x1 Conv (binary logits)
- Classification head: AdaptiveAvgPool2d → Dropout → Linear

### Attention U-Net

Same encoder/bottleneck/decoder, but each skip connection passes through
an **AttentionGate** that learns to emphasise spatially relevant features
and suppress irrelevant background regions before concatenation.

---

## Multi-Task Training

```
total_loss = seg_bce + seg_dice + classification_weight * cls_crossentropy
```

- **Segmentation**: BCEWithLogitsLoss + DiceLoss
- **Classification**: CrossEntropyLoss (37 breed classes)
- Both tasks share the encoder; segmentation and classification are
  trained jointly.

---

## Configuration

All hyperparameters are centralised in `config.py`. Key settings:

| Parameter                | Default |
|--------------------------|---------|
| IMAGE_SIZE               | 256     |
| BATCH_SIZE               | 8       |
| NUM_EPOCHS               | 30      |
| LEARNING_RATE            | 1e-3    |
| CLASSIFICATION_LOSS_WEIGHT | 1.0   |
| SEGMENTATION_THRESHOLD   | 0.5     |
| USE_AMP                  | True    |
| DEVICE                   | cuda    |

---

## Notes

- The dataset README specifies trimap values: **1=Foreground, 2=Background, 3=Boundary**. Boundary pixels are treated as foreground for binary segmentation.
- Images are resized to 256x256. Masks use nearest-neighbour interpolation to preserve binary values.
- Training augmentation: random horizontal flip, rotation, resized crop, brightness/contrast jitter.
- Validation and test use resize-only (no random augmentation).
- Mixed precision (AMP) is enabled by default for faster CUDA training.
