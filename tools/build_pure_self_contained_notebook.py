"""
Build 100% Pure Standalone Notebook for Kaggle.
No external script imports (no import train_unet, no import config, no git commands).
Everything executes inline inside notebook cells.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def code_cell(code):
    lines = code.strip().split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines]
    }

cells = []

# --- Title ---
cells.append(md_cell("""# CSE428 — Oxford-IIIT Pet: Multi-Task Segmentation + Breed Classification
**100% Standalone Self-Contained Notebook**

This notebook performs joint **binary pet segmentation** (foreground vs background) and **breed classification** (37 classes) using two architectures:
1. **Base U-Net** (shared encoder + segmentation decoder + classification head)
2. **Attention U-Net** (attention-gated skip connections + dual heads)

**This notebook is 100% self-contained:** all model architectures, loss functions, dataset loaders, metrics, and training loops are written directly in notebook cells below. No external `.py` scripts (`train_unet.py`, `config.py`) or `git clone` commands are used."""))

# --- 1. GPU Check ---
cells.append(md_cell("## 1. GPU Verification"))
cells.append(code_cell("!nvidia-smi"))

# --- 2. Imports & Config ---
cells.append(md_cell("## 2. Imports & Workspace Configuration"))
cells.append(code_cell("""import os
import sys
import time
import math
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # Pre-initialized for Python 3.12 compatibility
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Allow loading truncated JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Global Configuration ---
class Config:
    IMAGE_SIZE = 256
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    VAL_RATIO = 0.1
    RANDOM_SEED = 42
    CLASSIFICATION_LOSS_WEIGHT = 1.0
    SEGMENTATION_THRESHOLD = 0.5
    NUM_WORKERS = 2
    USE_AMP = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else (Path("/content") if Path("/content").is_dir() else Path.cwd())
    CHECKPOINT_DIR = WORK_DIR / "checkpoints"
    RESULTS_DIR = WORK_DIR / "results"

Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

print(f"Configuration Ready | Device: {Config.DEVICE} | AMP: {Config.USE_AMP}")
print(f"Checkpoints: {Config.CHECKPOINT_DIR} | Results: {Config.RESULTS_DIR}")"""))

# --- 3. Dataset Download ---
cells.append(md_cell("""## 3. Dataset Fetcher & Path Discovery
Automatically discovers dataset files or downloads official Oxford-IIIT Pet tarballs."""))
cells.append(code_cell("""def discover_dataset_paths() -> Tuple[Optional[Path], Optional[Path]]:
    search_roots = [
        Path.cwd(),
        Config.WORK_DIR,
        Config.WORK_DIR / "data",
        Path("/kaggle/input"),
        Path("/content"),
        Path("/content/data"),
    ]
    if Path("/kaggle/input").is_dir():
        try:
            search_roots.extend(sorted(p for p in Path("/kaggle/input").iterdir() if p.is_dir()))
        except Exception:
            pass

    for root in search_roots:
        if not root.is_dir():
            continue
        ann_candidates = [root / "annotations" / "annotations", root / "annotations", root / "data" / "annotations"]
        found_ann = None
        for candidate in ann_candidates:
            if (candidate / "trimaps").is_dir() and ((candidate / "list.txt").is_file() or (candidate / "trainval.txt").is_file()):
                found_ann = candidate
                break
        
        if found_ann is not None:
            img_candidates = [found_ann.parent / "images", found_ann.parent.parent / "images", root / "images", root / "data" / "images"]
            for img in img_candidates:
                if img.is_dir() and any(img.glob("*.jpg")):
                    return img, found_ann
    return None, None

def ensure_dataset() -> Tuple[Path, Path]:
    img_dir, ann_dir = discover_dataset_paths()
    if img_dir is not None and ann_dir is not None:
        print(f"Dataset discovered:\\n  Images: {img_dir}\\n  Annotations: {ann_dir}")
        return img_dir, ann_dir
    
    target_data_dir = Config.WORK_DIR / "data"
    target_data_dir.mkdir(parents=True, exist_ok=True)
    
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    def _download_and_extract(url: str, dest_dir: Path):
        tar_path = dest_dir / url.split("/")[-1]
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, tar_path)
        print(f"Extracting {tar_path.name} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        tar_path.unlink()

    _download_and_extract(images_url, target_data_dir)
    _download_and_extract(annotations_url, target_data_dir)
    
    img_dir, ann_dir = discover_dataset_paths()
    if img_dir is None or ann_dir is None:
        img_dir = target_data_dir / "images"
        ann_dir = target_data_dir / "annotations"
    print(f"Dataset ready:\\n  Images: {img_dir}\\n  Annotations: {ann_dir}")
    return img_dir, ann_dir

IMAGE_DIR, ANNOTATION_DIR = ensure_dataset()
TRIMAP_DIR = ANNOTATION_DIR / "trimaps\""""))

# --- 4. Dataset Loader ---
cells.append(md_cell("## 4. Dataset Loader & Split Processing"))
cells.append(code_cell("""def parse_annotation_file(filepath: Path) -> List[Dict]:
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            entries.append({
                "image_name": parts[0],
                "class_id": int(parts[1]) - 1,
                "species": int(parts[2]),
                "breed_id": int(parts[3]) - 1,
            })
    return entries

def get_splits(ann_dir: Path, img_dir: Path, trimap_dir: Path, val_ratio: float = 0.1, seed: int = 42):
    if not (ann_dir / "trainval.txt").is_file():
        for candidate in [ann_dir / "annotations", ann_dir.parent, ann_dir.parent / "annotations"]:
            if (candidate / "trainval.txt").is_file():
                ann_dir = candidate
                break
                
    trainval = parse_annotation_file(ann_dir / "trainval.txt")
    test = parse_annotation_file(ann_dir / "test.txt")
    all_entries = parse_annotation_file(ann_dir / "list.txt")
    
    class_names = sorted(list(set(e["image_name"].rsplit("_", 1)[0] for e in all_entries)))
    c2i = {name: idx for idx, name in enumerate(class_names)}
    i2c = {idx: name for name, idx in c2i.items()}
    
    breed_species = {}
    for e in all_entries:
        cname = e["image_name"].rsplit("_", 1)[0]
        breed_species[cname] = "Cat" if e["species"] == 1 else "Dog"

    rng = np.random.RandomState(seed)
    indices = np.arange(len(trainval))
    rng.shuffle(indices)
    
    n_val = max(1, int(len(trainval) * val_ratio))
    val_indices = set(indices[:n_val].tolist())
    train_indices = set(indices[n_val:].tolist())
    
    train_entries = [trainval[i] for i in sorted(train_indices)]
    val_entries = [trainval[i] for i in sorted(val_indices)]
    
    return train_entries, val_entries, test, c2i, i2c, breed_species

class OxfordPetDataset(Dataset):
    def __init__(self, img_dir: Path, trimap_dir: Path, entries: List[Dict], c2i: Dict[str, int], image_size: int = 256, is_train: bool = False):
        self.img_dir = img_dir
        self.trimap_dir = trimap_dir
        self.entries = entries
        self.c2i = c2i
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        name = entry["image_name"]
        
        img_path = self.img_dir / f"{name}.jpg"
        mask_path = self.trimap_dir / f"{name}.png"
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        img_np = np.array(image, dtype=np.float32) / 255.0
        mask_np = np.array(mask, dtype=np.int64)
        
        # Binary mask conversion: foreground (1) and boundary (3) -> 1.0, background (2) -> 0.0
        binary_mask = np.where((mask_np == 1) | (mask_np == 3), 1.0, 0.0).astype(np.float32)
        
        if self.is_train and random.random() > 0.5:
            img_np = np.fliplr(img_np).copy()
            binary_mask = np.fliplr(binary_mask).copy()
            
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()
        
        cname = name.rsplit("_", 1)[0]
        class_idx = self.c2i[cname]
        
        return img_tensor, mask_tensor, torch.tensor(class_idx, dtype=torch.long)

train_entries, val_entries, test_entries, c2i, i2c, breed_species = get_splits(
    ANNOTATION_DIR, IMAGE_DIR, TRIMAP_DIR, val_ratio=Config.VAL_RATIO, seed=Config.RANDOM_SEED
)

train_ds = OxfordPetDataset(IMAGE_DIR, TRIMAP_DIR, train_entries, c2i, image_size=Config.IMAGE_SIZE, is_train=True)
val_ds = OxfordPetDataset(IMAGE_DIR, TRIMAP_DIR, val_entries, c2i, image_size=Config.IMAGE_SIZE, is_train=False)
test_ds = OxfordPetDataset(IMAGE_DIR, TRIMAP_DIR, test_entries, c2i, image_size=Config.IMAGE_SIZE, is_train=False)

train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)

print(f"Splits Loaded | Train: {len(train_entries)} | Val: {len(val_entries)} | Test: {len(test_entries)} | Classes: {len(c2i)}")"""))

# --- 5. Exploration Grid ---
cells.append(md_cell("## 5. Dataset Exploration (3x3 Grid Overlay)"))
cells.append(code_cell("""plt.figure(figsize=(10, 10))
sample_indices = random.sample(range(len(train_entries)), 9)

for idx, sample_idx in enumerate(sample_indices):
    entry = train_entries[sample_idx]
    name = entry["image_name"]
    img = Image.open(IMAGE_DIR / f"{name}.jpg").convert("RGB").resize((256, 256))
    mask = Image.open(TRIMAP_DIR / f"{name}.png").resize((256, 256), Image.NEAREST)
    bin_mask = np.where((np.array(mask) == 1) | (np.array(mask) == 3), 1.0, 0.0)
    cname = name.rsplit("_", 1)[0]
    
    plt.subplot(3, 3, idx + 1)
    plt.imshow(img)
    plt.imshow(bin_mask, alpha=0.4, cmap="jet")
    plt.title(f"{cname} ({breed_species[cname]})", fontsize=10)
    plt.axis("off")
    
plt.tight_layout()
plt.savefig(Config.RESULTS_DIR / "exploration_samples.png", bbox_inches="tight")
plt.show()"""))

# --- 6. Architectures ---
cells.append(md_cell("## 6. Model Architectures: Base U-Net & Attention U-Net"))
cells.append(code_cell("""class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.mpconv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

class AttnUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.ag = AttentionGate(F_g=in_ch // 2, F_l=in_ch // 2, F_int=in_ch // 4)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2_gated = self.ag(g=x1, x=x2)
        x = torch.cat([x2_gated, x1], dim=1)
        return self.conv(x)

class BaseUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=37, base_features=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = DownBlock(base_features, base_features * 2)
        self.down2 = DownBlock(base_features * 2, base_features * 4)
        self.down3 = DownBlock(base_features * 4, base_features * 8)
        self.down4 = DownBlock(base_features * 8, base_features * 16)
        
        self.up1 = UpBlock(base_features * 16, base_features * 8)
        self.up2 = UpBlock(base_features * 8, base_features * 4)
        self.up3 = UpBlock(base_features * 4, base_features * 2)
        self.up4 = UpBlock(base_features * 2, base_features)
        self.outc = nn.Conv2d(base_features, 1, kernel_size=1)
        
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(base_features * 16, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x_seg = self.up1(x5, x4); x_seg = self.up2(x_seg, x3); x_seg = self.up3(x_seg, x2); x_seg = self.up4(x_seg, x1)
        return self.outc(x_seg), self.cls_head(x5)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=37, base_features=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = DownBlock(base_features, base_features * 2)
        self.down2 = DownBlock(base_features * 2, base_features * 4)
        self.down3 = DownBlock(base_features * 4, base_features * 8)
        self.down4 = DownBlock(base_features * 8, base_features * 16)
        
        self.up1 = AttnUpBlock(base_features * 16, base_features * 8)
        self.up2 = AttnUpBlock(base_features * 8, base_features * 4)
        self.up3 = AttnUpBlock(base_features * 4, base_features * 2)
        self.up4 = AttnUpBlock(base_features * 2, base_features)
        self.outc = nn.Conv2d(base_features, 1, kernel_size=1)
        
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(base_features * 16, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x_seg = self.up1(x5, x4); x_seg = self.up2(x_seg, x3); x_seg = self.up3(x_seg, x2); x_seg = self.up4(x_seg, x1)
        return self.outc(x_seg), self.cls_head(x5)"""))

# --- 7. Losses & Metrics ---
cells.append(md_cell("## 7. Multi-Task Loss & Evaluation Metrics"))
cells.append(code_cell("""class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs * targets_flat).sum()
        return 1.0 - ((2.0 * intersection + self.smooth) / (probs.sum() + targets_flat.sum() + self.smooth))

class MultiTaskLoss(nn.Module):
    def __init__(self, cls_weight=1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, seg_logits, cls_logits, seg_targets, cls_targets):
        bce_l = self.bce(seg_logits, seg_targets)
        dice_l = self.dice(seg_logits, seg_targets)
        cls_l = self.ce(cls_logits, cls_targets)
        return bce_l + dice_l + (self.cls_weight * cls_l), bce_l + dice_l, cls_l

def compute_metrics(seg_logits, cls_logits, seg_targets, cls_targets):
    probs = torch.sigmoid(seg_logits)
    preds_mask = (probs > 0.5).float()
    
    intersection = (preds_mask * seg_targets).sum().item()
    union = (preds_mask + seg_targets).clamp(0, 1).sum().item()
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2.0 * intersection + 1e-6) / (preds_mask.sum().item() + seg_targets.sum().item() + 1e-6)
    
    cls_preds = torch.argmax(cls_logits, dim=1).cpu().numpy()
    cls_true = cls_targets.cpu().numpy()
    
    acc = accuracy_score(cls_true, cls_preds)
    f1 = f1_score(cls_true, cls_preds, average="macro", zero_division=0)
    
    return iou, dice, acc, f1"""))

# --- 8. Inline Cell 7: Train Base U-Net ---
cells.append(md_cell("""## 8. Train Base U-Net Model (100% Inline Execution)
Shared encoder + Segmentation Decoder + 37-class Breed Classifier.
Learning rate scheduling and best validation IoU checkpointing enabled."""))

cells.append(code_cell("""base_unet = BaseUNet(in_channels=3, num_classes=len(c2i), base_features=64).to(Config.DEVICE)
optimizer = torch.optim.Adam(base_unet.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
criterion = MultiTaskLoss(cls_weight=Config.CLASSIFICATION_LOSS_WEIGHT)
scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)

try:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
except TypeError:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

best_val_iou = 0.0
ckpt_path = Config.CHECKPOINT_DIR / "best_unet.pth"
csv_path = Config.RESULTS_DIR / "unet_history.csv"

unet_history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "val_acc": [], "val_f1": []}

print(f"--- Starting Base U-Net Training on {Config.DEVICE} ---")
start_time = time.time()

for epoch in range(1, Config.NUM_EPOCHS + 1):
    base_unet.train()
    running_loss = 0.0
    for imgs, masks, classes in tqdm(train_loader, desc=f"Base U-Net Epoch {epoch:02d}/{Config.NUM_EPOCHS:02d}"):
        imgs, masks, classes = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), classes.to(Config.DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            seg_out, cls_out = base_unet(imgs)
            loss, _, _ = criterion(seg_out, cls_out, masks, classes)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        
    train_loss = running_loss / len(train_loader.dataset)
    
    base_unet.eval()
    val_loss = 0.0
    val_ious, val_dices, val_accs, val_f1s = [], [], [], []
    with torch.no_grad():
        for imgs, masks, classes in val_loader:
            imgs, masks, classes = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), classes.to(Config.DEVICE)
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                seg_out, cls_out = base_unet(imgs)
                loss, _, _ = criterion(seg_out, cls_out, masks, classes)
            val_loss += loss.item() * imgs.size(0)
            iou, dice, acc, f1 = compute_metrics(seg_out, cls_out, masks, classes)
            val_ious.append(iou); val_dices.append(dice); val_accs.append(acc); val_f1s.append(f1)
            
    val_loss = val_loss / len(val_loader.dataset)
    mean_iou = float(np.mean(val_ious))
    mean_acc = float(np.mean(val_accs))
    
    scheduler.step(mean_iou)
    
    unet_history["epoch"].append(epoch)
    unet_history["train_loss"].append(train_loss)
    unet_history["val_loss"].append(val_loss)
    unet_history["val_iou"].append(mean_iou)
    unet_history["val_dice"].append(float(np.mean(val_dices)))
    unet_history["val_acc"].append(mean_acc)
    unet_history["val_f1"].append(float(np.mean(val_f1s)))
    
    print(f"Epoch {epoch:02d}/{Config.NUM_EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {mean_iou:.4f} | Val Acc: {mean_acc:.4f}")
    
    if mean_iou > best_val_iou:
        best_val_iou = mean_iou
        torch.save({"epoch": epoch, "model_state_dict": base_unet.state_dict(), "val_iou": mean_iou}, ckpt_path)
        print(f"  --> Saved new best checkpoint to {ckpt_path.name}")

pd.DataFrame(unet_history).to_csv(csv_path, index=False)
elapsed = time.time() - start_time
print(f"\\nBase U-Net training completed in {elapsed/60:.1f} minutes.")"""))

# --- 9. Inline Cell 8: Train Attention U-Net ---
cells.append(md_cell("""## 9. Train Attention U-Net Model (100% Inline Execution)
Attention-gated skip connections + dual task heads."""))

cells.append(code_cell("""attn_unet = AttentionUNet(in_channels=3, num_classes=len(c2i), base_features=64).to(Config.DEVICE)
optimizer = torch.optim.Adam(attn_unet.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
criterion = MultiTaskLoss(cls_weight=Config.CLASSIFICATION_LOSS_WEIGHT)
scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)

try:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
except TypeError:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

best_val_iou = 0.0
ckpt_path = Config.CHECKPOINT_DIR / "best_attention_unet.pth"
csv_path = Config.RESULTS_DIR / "attention_unet_history.csv"

attn_history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "val_acc": [], "val_f1": []}

print(f"--- Starting Attention U-Net Training on {Config.DEVICE} ---")
start_time = time.time()

for epoch in range(1, Config.NUM_EPOCHS + 1):
    attn_unet.train()
    running_loss = 0.0
    for imgs, masks, classes in tqdm(train_loader, desc=f"Attn U-Net Epoch {epoch:02d}/{Config.NUM_EPOCHS:02d}"):
        imgs, masks, classes = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), classes.to(Config.DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            seg_out, cls_out = attn_unet(imgs)
            loss, _, _ = criterion(seg_out, cls_out, masks, classes)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        
    train_loss = running_loss / len(train_loader.dataset)
    
    attn_unet.eval()
    val_loss = 0.0
    val_ious, val_dices, val_accs, val_f1s = [], [], [], []
    with torch.no_grad():
        for imgs, masks, classes in val_loader:
            imgs, masks, classes = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), classes.to(Config.DEVICE)
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                seg_out, cls_out = attn_unet(imgs)
                loss, _, _ = criterion(seg_out, cls_out, masks, classes)
            val_loss += loss.item() * imgs.size(0)
            iou, dice, acc, f1 = compute_metrics(seg_out, cls_out, masks, classes)
            val_ious.append(iou); val_dices.append(dice); val_accs.append(acc); val_f1s.append(f1)
            
    val_loss = val_loss / len(val_loader.dataset)
    mean_iou = float(np.mean(val_ious))
    mean_acc = float(np.mean(val_accs))
    
    scheduler.step(mean_iou)
    
    attn_history["epoch"].append(epoch)
    attn_history["train_loss"].append(train_loss)
    attn_history["val_loss"].append(val_loss)
    attn_history["val_iou"].append(mean_iou)
    attn_history["val_dice"].append(float(np.mean(val_dices)))
    attn_history["val_acc"].append(mean_acc)
    attn_history["val_f1"].append(float(np.mean(val_f1s)))
    
    print(f"Epoch {epoch:02d}/{Config.NUM_EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {mean_iou:.4f} | Val Acc: {mean_acc:.4f}")
    
    if mean_iou > best_val_iou:
        best_val_iou = mean_iou
        torch.save({"epoch": epoch, "model_state_dict": attn_unet.state_dict(), "val_iou": mean_iou}, ckpt_path)
        print(f"  --> Saved new best checkpoint to {ckpt_path.name}")

pd.DataFrame(attn_history).to_csv(csv_path, index=False)
elapsed = time.time() - start_time
print(f"\\nAttention U-Net training completed in {elapsed/60:.1f} minutes.")"""))

# --- 10. Quantitative Evaluation Table ---
cells.append(md_cell("## 10. Quantitative Metric Evaluation & Summary Table"))
cells.append(code_cell("""def evaluate_checkpoint(model_class, ckpt_name, loader):
    ckpt_path = Config.CHECKPOINT_DIR / f"best_{ckpt_name}.pth"
    model = model_class(in_channels=3, num_classes=len(c2i), base_features=64).to(Config.DEVICE)
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    ious, dices, accs, precs, recs, f1s = [], [], [], [], [], []
    with torch.no_grad():
        for imgs, masks, classes in loader:
            imgs, masks, classes = imgs.to(Config.DEVICE), masks.to(Config.DEVICE), classes.to(Config.DEVICE)
            seg_out, cls_out = model(imgs)
            probs = torch.sigmoid(seg_out)
            preds_mask = (probs > 0.5).float()
            
            for i in range(imgs.size(0)):
                intersection = (preds_mask[i] * masks[i]).sum().item()
                union = (preds_mask[i] + masks[i]).clamp(0, 1).sum().item()
                ious.append((intersection + 1e-6) / (union + 1e-6))
                dices.append((2.0 * intersection + 1e-6) / (preds_mask[i].sum().item() + masks[i].sum().item() + 1e-6))
                
            cls_preds = torch.argmax(cls_out, dim=1).cpu().numpy()
            cls_true = classes.cpu().numpy()
            
            accs.append(accuracy_score(cls_true, cls_preds))
            precs.append(precision_score(cls_true, cls_preds, average="macro", zero_division=0))
            recs.append(recall_score(cls_true, cls_preds, average="macro", zero_division=0))
            f1s.append(f1_score(cls_true, cls_preds, average="macro", zero_division=0))
            
    return {
        "mIoU": np.mean(ious), "Dice": np.mean(dices),
        "Accuracy": np.mean(accs), "Precision": np.mean(precs),
        "Recall": np.mean(recs), "F1": np.mean(f1s)
    }

results_summary = []
for name, mclass, ckpt_name in [("Base U-Net", BaseUNet, "unet"), ("Attention U-Net", AttentionUNet, "attention_unet")]:
    for split_name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        m = evaluate_checkpoint(mclass, ckpt_name, loader)
        results_summary.append({"Model": name, "Split": split_name, **m})

summary_df = pd.DataFrame(results_summary)
print("\\n--- Final Comparative Performance Summary ---")
print(summary_df.to_string(index=False))
summary_df.to_csv(Config.RESULTS_DIR / "final_model_comparison.csv", index=False)"""))

# --- 11. Visual Predictions Demo ---
cells.append(md_cell("## 11. Visual Predictions Overlay"))
cells.append(code_cell("""def visualize_predictions(sample_idx=0):
    entry = test_entries[sample_idx]
    name = entry["image_name"]
    cname = name.rsplit("_", 1)[0]
    
    img = Image.open(IMAGE_DIR / f"{name}.jpg").convert("RGB").resize((256, 256))
    mask = Image.open(TRIMAP_DIR / f"{name}.png").resize((256, 256), Image.NEAREST)
    bin_mask = np.where((np.array(mask) == 1) | (np.array(mask) == 3), 1.0, 0.0)
    
    unet = BaseUNet(num_classes=len(c2i)).to(Config.DEVICE)
    if (Config.CHECKPOINT_DIR / "best_unet.pth").exists():
        unet.load_state_dict(torch.load(Config.CHECKPOINT_DIR / "best_unet.pth", map_location=Config.DEVICE)["model_state_dict"])
    unet.eval()
    
    attn = AttentionUNet(num_classes=len(c2i)).to(Config.DEVICE)
    if (Config.CHECKPOINT_DIR / "best_attention_unet.pth").exists():
        attn.load_state_dict(torch.load(Config.CHECKPOINT_DIR / "best_attention_unet.pth", map_location=Config.DEVICE)["model_state_dict"])
    attn.eval()
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (np.array(img, dtype=np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(Config.DEVICE)
    
    with torch.no_grad():
        u_seg, u_cls = unet(tensor)
        a_seg, a_cls = attn(tensor)
        
    u_mask = (torch.sigmoid(u_seg)[0, 0] > 0.5).cpu().numpy()
    a_mask = (torch.sigmoid(a_seg)[0, 0] > 0.5).cpu().numpy()
    
    u_pred_class = i2c[torch.argmax(u_cls, dim=1).item()]
    a_pred_class = i2c[torch.argmax(a_cls, dim=1).item()]
    
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1); plt.imshow(img); plt.title(f"Input: {cname}"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(img); plt.imshow(bin_mask, alpha=0.4, cmap="jet"); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(img); plt.imshow(u_mask, alpha=0.4, cmap="jet"); plt.title(f"Base U-Net\\nPred: {u_pred_class}"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(img); plt.imshow(a_mask, alpha=0.4, cmap="jet"); plt.title(f"Attn U-Net\\nPred: {a_pred_class}"); plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(Config.RESULTS_DIR / "demo_predictions.png", bbox_inches="tight")
    plt.show()

visualize_predictions(sample_idx=5)"""))

nb_json = {
    "cells": cells,
    "metadata": {"language_info": {"name": "python"}, "accelerator": "GPU"},
    "nbformat": 4, "nbformat_minor": 2
}

targets = [
    ROOT / "multi-task-pet-segmentation.ipynb",
    ROOT / "notebooks" / "main" / "multi-task-pet-segmentation.ipynb",
]

for t in targets:
    t.parent.mkdir(parents=True, exist_ok=True)
    with open(t, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=1)
    print(f"Saved pure standalone notebook to: {t}")

print("Build complete!")
