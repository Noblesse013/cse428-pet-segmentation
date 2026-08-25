# CSE428 — Multi-Task Pet Segmentation & Breed Classification

Multi-task deep-learning project using the **Oxford-IIIT Pet dataset**.
The model jointly learns **binary pet segmentation** (foreground vs. background)
and **breed classification** (37 classes, Cat/Dog).

Two architectures are implemented and compared:

1. **Base U-Net** + classification head
2. **Attention U-Net** + classification head

Both share a common encoder and train segmentation + classification together.

---

## Run it on a free GPU (Colab / Kaggle)

No local GPU needed. The notebook clones this repo, downloads the dataset,
trains both models, evaluates them and renders the demo figures.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Noblesse013/cse428-pet-segmentation/blob/master/notebooks/CSE428_Pet_MultiTask_Colab_Kaggle.ipynb)

**Colab** — click the badge, then `Runtime -> Change runtime type -> T4 GPU`, then `Runtime -> Run all`.

**Kaggle** — `New Notebook -> File -> Import Notebook`, upload
`notebooks/CSE428_Pet_MultiTask_Colab_Kaggle.ipynb`, then in the sidebar set
`Accelerator -> GPU T4 x2` and `Internet -> On`, and `Run All`.

Set `config.NUM_EPOCHS = 5` in section 5 for a quick end-to-end check before
committing to the full 30-epoch run (~45-60 min per model on a T4).

On Kaggle you can skip the download entirely by attaching an Oxford-IIIT Pet
dataset with `+ Add Input` — `config.py` finds anything under `/kaggle/input`
that contains `images/` and `annotations/trimaps/`.

---

## Requirements

- **NVIDIA CUDA-enabled GPU** and a CUDA-enabled PyTorch installation are **required**
  for the training / evaluation scripts; they raise a clear error if no CUDA GPU is
  found. Use the Colab/Kaggle notebook above if you don't have one.
- Python 3.9+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The Oxford-IIIT Pet dataset is not tracked in git (~800 MB). Either extract
`images/` and `annotations/` into the project root:

```
images/           # JPG pet images
annotations/      # Annotation files (list.txt, trainval.txt, test.txt, trimaps/)
```

...or let the helper fetch it for you:

```python
from src.data_setup import ensure_dataset
image_dir, annotation_dir = ensure_dataset()
```

`config.py` discovers the dataset at import time, searching the project root,
`data/`, `/content` (Colab) and every directory under `/kaggle/input` (Kaggle),
so the nested layout above and a Kaggle attached input both work unchanged.
Override the search with the `PET_IMAGE_DIR` / `PET_ANNOTATION_DIR` env vars.

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
│   ├── data_setup.py        # Dataset download/discovery for cloud notebooks
│   ├── amp_compat.py        # Mixed-precision shim across PyTorch versions
│   └── models/
│       ├── blocks.py        # DoubleConv, DownBlock, UpBlock
│       ├── unet.py          # Base U-Net + classifier
│       ├── attention_blocks.py  # AttentionGate
│       └── attention_unet.py # Attention U-Net + classifier
│
├── notebooks/
│   └── CSE428_Pet_MultiTask_Colab_Kaggle.ipynb   # Colab / Kaggle runner
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

| Parameter                  | Default                | Env-var override         |
|----------------------------|------------------------|--------------------------|
| IMAGE_SIZE                 | 256                    | `PET_IMAGE_SIZE`         |
| BATCH_SIZE                 | 8 local / 16 on cloud  | `PET_BATCH_SIZE`         |
| NUM_EPOCHS                 | 30                     | `PET_NUM_EPOCHS`         |
| LEARNING_RATE              | 1e-3                   | `PET_LEARNING_RATE`      |
| CLASSIFICATION_LOSS_WEIGHT | 1.0                    | `PET_CLS_LOSS_WEIGHT`    |
| SEGMENTATION_THRESHOLD     | 0.5                    | `PET_SEG_THRESHOLD`      |
| NUM_WORKERS                | 0 on Windows, else 2   | `PET_NUM_WORKERS`        |
| USE_AMP                    | True when CUDA present | `PET_USE_AMP`            |
| DEVICE                     | cuda if available      | -                        |

Every value can also be set from a notebook before training:

```python
import config
config.NUM_EPOCHS = 10
history = train_unet.main(num_epochs=10, batch_size=16)   # or pass directly
```

`config.describe_environment()` prints the resolved environment, device, paths
and hyperparameters — worth running first in any new session.

---

## Notes

- The dataset README specifies trimap values: **1=Foreground, 2=Background, 3=Boundary**. Boundary pixels are treated as foreground for binary segmentation.
- Images are resized to 256x256. Masks use nearest-neighbour interpolation to preserve binary values.
- Training augmentation: random horizontal flip, rotation, resized crop, brightness/contrast jitter.
- Validation and test use resize-only (no random augmentation).
- Mixed precision (AMP) is enabled by default for faster CUDA training, and is
  automatically disabled on CPU. `src/amp_compat.py` selects the `torch.amp` or
  legacy `torch.cuda.amp` API depending on the installed PyTorch version.
- `train_unet.main()` / `train_attention_unet.main()` accept keyword overrides and
  return the training-history dict, so they are usable from a notebook as well as
  from the command line.
