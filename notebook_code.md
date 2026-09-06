In [10]:
!nvidia-smi

---

In [11]:
import os, sys, shutil, subprocess
from pathlib import Path

REPO_URL  = "https://github.com/Noblesse013/cse428-pet-segmentation.git"
REPO_NAME = "cse428-pet-segmentation"

IN_COLAB  = "google.colab" in sys.modules
IN_KAGGLE = Path("/kaggle/input").is_dir()
WORK = Path("/kaggle/working") if IN_KAGGLE else (Path("/content") if IN_COLAB else Path.cwd())


def is_project(path):
    """A project root has config.py and the src package."""
    try:
        return (path / "config.py").is_file() and (path / "src" / "models").is_dir()
    except OSError:
        return False


def search_paths():
    """Everywhere the project code could plausibly already be."""
    yield Path.cwd()
    yield WORK / REPO_NAME
    yield WORK
    yield Path.cwd().parent
    # Kaggle: code uploaded as a Dataset lands under /kaggle/input/<slug>/ ...
    # possibly one folder deeper, if the zip had a top-level folder inside it.
    if IN_KAGGLE:
        for dataset in sorted(Path("/kaggle/input").iterdir()):
            if not dataset.is_dir():
                continue
            yield dataset
            try:
                for child in sorted(c for c in dataset.iterdir() if c.is_dir()):
                    yield child
            except OSError:
                continue


PROJECT_DIR = next((p.resolve() for p in search_paths() if is_project(p)), None)

if PROJECT_DIR is not None:
    print("Found the project code at:", PROJECT_DIR)
else:
    if not REPO_URL:
        raise SystemExit("No project code found, and REPO_URL is empty.")
    target = WORK / REPO_NAME
    print("Cloning", REPO_URL, "->", target)
    result = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)],
                            capture_output=True, text=True)
    print(result.stdout or result.stderr)
    if not is_project(target):
        raise SystemExit("""Could not find or clone the project.

Kaggle: cloning needs Internet -- sidebar > Session options > Internet: On
        (that requires a phone-verified account). If you cannot enable it,
        upload the code as a Dataset and attach it with '+ Add Input';
        this cell searches /kaggle/input and will find it.
Colab : upload a zip via the files pane, then run
        !unzip -q yourzip.zip -d /content

Then re-run this cell.""")
    PROJECT_DIR = target.resolve()

# /kaggle/input is mounted read-only. The code tree is small, so copy it
# somewhere writable and work from there; the dataset stays where it is.
if IN_KAGGLE and str(PROJECT_DIR).startswith("/kaggle/input"):
    writable = WORK / REPO_NAME
    if writable.exists():
        shutil.rmtree(writable)
    shutil.copytree(PROJECT_DIR, writable,
                    ignore=shutil.ignore_patterns("images", "annotations", "data",
                                                  "__pycache__", ".git"))
    print("Copied read-only code ->", writable)
    PROJECT_DIR = writable

os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

print()
print("Working directory:", Path.cwd())
print("Contents:", sorted(p.name for p in PROJECT_DIR.iterdir()
                          if not p.name.startswith("."))[:15])


---

In [12]:
import importlib, subprocess, sys

required = {
    "torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
    "pandas": "pandas", "matplotlib": "matplotlib", "PIL": "Pillow",
    "sklearn": "scikit-learn", "tqdm": "tqdm",
}

missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
else:
    print("All dependencies already present.")

import torch
print(f"torch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}")

---

In [13]:
import config
from src.data_setup import ensure_dataset, dataset_summary, free_disk_space

print(free_disk_space(), "\n")
image_dir, annotation_dir = ensure_dataset()
print()
print(dataset_summary())

---

In [14]:
import config

# ---- edit these ----------------------------------------------------------
config.NUM_EPOCHS  = 5      # raise to 30 for the full run
config.BATCH_SIZE  = 16     # 16 fits a 16 GB T4 at 256px; drop to 8 if OOM
config.IMAGE_SIZE  = 256
config.LEARNING_RATE = 1e-3
config.NUM_WORKERS = 2      # Colab/Kaggle VMs have 2-4 cores
# --------------------------------------------------------------------------

config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(config.describe_environment())
config.validate_cuda()   # fails loudly here rather than 20 minutes into training
print("\nGPU confirmed.")

---

In [15]:
USE_DRIVE = False   # set True to persist checkpoints/results to Google Drive

if USE_DRIVE and "google.colab" in __import__("sys").modules:
    from google.colab import drive
    from pathlib import Path
    drive.mount("/content/drive")
    out = Path("/content/drive/MyDrive/cse428_pet")
    config.CHECKPOINT_DIR = out / "checkpoints"
    config.RESULTS_DIR = out / "results"
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Outputs ->", out)
else:
    print(f"Outputs stay in the session:\n  {config.CHECKPOINT_DIR}\n  {config.RESULTS_DIR}")

---

In [16]:
from IPython.display import Image as ShowImage, display

from src.dataset import get_splits, OxfordPetDataset
from src.transforms import get_eval_transform
from src.visualization import show_random_samples

train_entries, val_entries, test_entries, class_to_idx, idx_to_class, breed_species = get_splits(
    config.ANNOTATION_DIR, config.IMAGE_DIR, config.TRIMAP_DIR,
    val_ratio=config.VAL_RATIO, seed=config.RANDOM_SEED,
)
NUM_CLASSES = len(class_to_idx)

print(f"Train      : {len(train_entries)}")
print(f"Validation : {len(val_entries)}")
print(f"Test       : {len(test_entries)}")
print(f"Classes    : {NUM_CLASSES}")

cats = sum(1 for name in breed_species.values() if name == "Cat")
print(f"Breeds     : {cats} cat, {len(breed_species) - cats} dog")

explore_ds = OxfordPetDataset(
    config.IMAGE_DIR, config.TRIMAP_DIR, train_entries + val_entries, class_to_idx,
    image_size=config.IMAGE_SIZE, transform=get_eval_transform(config.IMAGE_SIZE), mode="val",
)

grid_path = str(config.RESULTS_DIR / "exploration_samples.png")
show_random_samples(explore_ds, n=9, save_path=grid_path)
display(ShowImage(filename=grid_path, width=900))

---

In [17]:
import time
import train_unet

start = time.time()
unet_history = train_unet.main(
    num_epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    image_size=config.IMAGE_SIZE,
    num_workers=config.NUM_WORKERS,
    checkpoint_dir=config.CHECKPOINT_DIR,
    results_dir=config.RESULTS_DIR,
)
print(f"\nBase U-Net finished in {(time.time() - start) / 60:.1f} min")

---

In [18]:
from IPython.display import Image as ShowImage, display

for plot in ("unet_loss_curves.png", "unet_seg_curves.png", "unet_cls_curves.png"):
    path = config.RESULTS_DIR / plot
    if path.exists():
        display(ShowImage(filename=str(path), width=1000))

---

In [19]:
import time
import train_attention_unet

start = time.time()
attn_history = train_attention_unet.main(
    num_epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    image_size=config.IMAGE_SIZE,
    num_workers=config.NUM_WORKERS,
    checkpoint_dir=config.CHECKPOINT_DIR,
    results_dir=config.RESULTS_DIR,
)
print(f"\nAttention U-Net finished in {(time.time() - start) / 60:.1f} min")

---

In [20]:
from IPython.display import Image as ShowImage, display

for plot in ("attention_unet_loss_curves.png", "attention_unet_seg_curves.png",
             "attention_unet_cls_curves.png"):
    path = config.RESULTS_DIR / plot
    if path.exists():
        display(ShowImage(filename=str(path), width=1000))

---

In [21]:
import torch
from torch.utils.data import DataLoader

from src.dataset import OxfordPetDataset
from src.transforms import get_eval_transform
from src.models.unet import BaseUNet
from src.models.attention_unet import AttentionUNet
from src.evaluation import full_evaluation, print_metrics_table

eval_tf = get_eval_transform(config.IMAGE_SIZE)


def make_loader(entries):
    ds = OxfordPetDataset(config.IMAGE_DIR, config.TRIMAP_DIR, entries, class_to_idx,
                          image_size=config.IMAGE_SIZE, transform=eval_tf, mode="val")
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                      num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)


loaders = {
    "train": make_loader(train_entries),
    "val": make_loader(val_entries),
    "test": make_loader(test_entries),
}


def load_checkpoint(model_key):
    cls = BaseUNet if model_key == "unet" else AttentionUNet
    ckpt_path = config.CHECKPOINT_DIR / f"best_{model_key}.pth"
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model = cls(in_channels=3, num_classes=NUM_CLASSES, base_features=64)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(config.DEVICE).eval()


all_metrics = {}
for key, display_name in (("unet", "BASE U-NET"), ("attention_unet", "ATTENTION U-NET")):
    model = load_checkpoint(key)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{display_name}: {params:,} parameters")
    all_metrics[key] = {split: full_evaluation(model, loader, config.DEVICE)
                        for split, loader in loaders.items()}
    print_metrics_table(display_name, all_metrics[key]["train"],
                        all_metrics[key]["val"], all_metrics[key]["test"])
    del model
    torch.cuda.empty_cache()

---

In [22]:
import pandas as pd

rows = []
labels = [("IoU", "iou"), ("Dice", "dice"), ("Pixel Accuracy", "pixel_accuracy"),
          ("Cls Accuracy", "cls_accuracy"), ("Cls Precision", "cls_precision"),
          ("Cls Recall", "cls_recall"), ("Cls F1", "cls_f1")]

for label, key in labels:
    base = all_metrics["unet"]["test"][key]
    attn = all_metrics["attention_unet"]["test"][key]
    rows.append({
        "Metric": label,
        "Base U-Net": round(base, 4),
        "Attention U-Net": round(attn, 4),
        "Delta": round(attn - base, 4),
    })

comparison = pd.DataFrame(rows)
comparison.to_csv(config.RESULTS_DIR / "model_comparison_test.csv", index=False)
display(comparison)

---

In [23]:
from IPython.display import Image as ShowImage, display

from demo import predict_by_index
from src.visualization import show_prediction

SAMPLE_INDICES = [12, 145, 900, 2100]
MODEL_KEY = "attention_unet"      # or "unet"

full_ds = OxfordPetDataset(
    config.IMAGE_DIR, config.TRIMAP_DIR,
    train_entries + val_entries + test_entries, class_to_idx,
    image_size=config.IMAGE_SIZE, transform=eval_tf, mode="val",
)

model = load_checkpoint(MODEL_KEY)
for index in SAMPLE_INDICES:
    if index >= len(full_ds):
        print(f"index {index} out of range (0-{len(full_ds) - 1}) — skipped")
        continue
    result = predict_by_index(index, model, full_ds, config.DEVICE, idx_to_class,
                              threshold=config.SEGMENTATION_THRESHOLD)
    out_path = str(config.RESULTS_DIR / f"demo_{MODEL_KEY}_{index}.png")
    show_prediction(result["image"], result["true_mask"], result["pred_mask"],
                    result["true_breed"], result["pred_breed"], result["true_species"],
                    result["iou"], save_path=out_path)
    display(ShowImage(filename=out_path, width=1000))

del model
torch.cuda.empty_cache()

---

In [24]:
import shutil, sys
from pathlib import Path

INCLUDE_CHECKPOINTS = False

staging = Path("/tmp/cse428_outputs") if not config.IN_KAGGLE else Path("/kaggle/working/_bundle")
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)

shutil.copytree(config.RESULTS_DIR, staging / "results", dirs_exist_ok=True)
if INCLUDE_CHECKPOINTS:
    shutil.copytree(config.CHECKPOINT_DIR, staging / "checkpoints", dirs_exist_ok=True)

archive = shutil.make_archive(str(staging.parent / "cse428_outputs"), "zip", staging)
print("Bundle:", archive, f"({Path(archive).stat().st_size / 1e6:.1f} MB)")
print("\nFiles:")
for item in sorted(staging.rglob("*")):
    if item.is_file():
        print(" ", item.relative_to(staging))

if "google.colab" in sys.modules:
    from google.colab import files
    files.download(archive)