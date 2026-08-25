"""
Central configuration for the Oxford-IIIT Pet multi-task project.

Works unchanged on three environments:

    * local machine   (Windows / Linux, repo checkout)
    * Google Colab    (/content/..., free T4 GPU)
    * Kaggle Notebook (/kaggle/input read-only data, /kaggle/working outputs)

Nothing here is hard-coded to one machine: dataset directories are discovered
at import time, and every hyperparameter can be overridden with an environment
variable (handy inside a notebook, where editing this file is awkward):

    import os
    os.environ["PET_BATCH_SIZE"] = "16"
    os.environ["PET_NUM_EPOCHS"] = "15"
    import config          # then read the overrides
"""

import os
import sys
from pathlib import Path

import torch

# -- Environment detection --------------------------------------------------
IN_COLAB = "google.colab" in sys.modules or Path("/content").is_dir()
IN_KAGGLE = Path("/kaggle/input").is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IN_NOTEBOOK_CLOUD = IN_COLAB or IN_KAGGLE


def _env(name: str, default, cast=str):
    """Read an env-var override, falling back to *default*."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


# -- Project root -----------------------------------------------------------
# __file__ always exists for this module, even when imported from a notebook.
PROJECT_ROOT = Path(__file__).resolve().parent

# Where results/checkpoints are written.  Kaggle only lets you write to
# /kaggle/working, and only files there survive as notebook outputs.
if IN_KAGGLE:
    OUTPUT_ROOT = Path(_env("PET_OUTPUT_ROOT", "/kaggle/working"))
else:
    OUTPUT_ROOT = Path(_env("PET_OUTPUT_ROOT", str(PROJECT_ROOT)))

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
RESULTS_DIR = OUTPUT_ROOT / "results"


# -- Dataset discovery ------------------------------------------------------
def _looks_like_annotation_dir(path: Path) -> bool:
    """True if *path* holds trimaps/ plus the official split files."""
    return (path / "trimaps").is_dir() and (path / "list.txt").is_file()


def _find_annotation_dir(base: Path, max_depth: int = 3):
    """Breadth-first search under *base* for the annotations directory."""
    if not base.is_dir():
        return None
    frontier = [(base, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        try:
            if _looks_like_annotation_dir(current):
                return current
            if depth < max_depth:
                frontier.extend(
                    (child, depth + 1)
                    for child in sorted(current.iterdir())
                    if child.is_dir() and not child.name.startswith(".")
                )
        except (PermissionError, OSError):
            continue
    return None


def _find_image_dir(base: Path, annotation_dir: Path, max_depth: int = 3):
    """Find the folder of .jpg pet photos near *base* / *annotation_dir*."""
    candidates = [
        annotation_dir.parent / "images",
        annotation_dir.parent.parent / "images",
        base / "images",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate

    # Fall back to a search for any directory named "images" holding jpgs.
    if not base.is_dir():
        return None
    frontier = [(base, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        try:
            if current.name == "images" and any(current.glob("*.jpg")):
                return current
            if depth < max_depth:
                frontier.extend(
                    (child, depth + 1)
                    for child in sorted(current.iterdir())
                    if child.is_dir() and not child.name.startswith(".")
                )
        except (PermissionError, OSError):
            continue
    return None


def _dataset_search_roots():
    """Candidate roots, most-specific first, for every supported environment."""
    roots = [PROJECT_ROOT, PROJECT_ROOT / "data", Path.cwd()]
    if IN_KAGGLE:
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.is_dir():
            try:
                roots.extend(sorted(p for p in kaggle_input.iterdir() if p.is_dir()))
            except OSError:
                pass
        roots.append(Path("/kaggle/working/data"))
        roots.append(Path("/kaggle/working"))
    if IN_COLAB:
        roots.extend([Path("/content/data"), Path("/content")])
    return roots


def discover_dataset():
    """Locate (image_dir, annotation_dir).  Returns (None, None) if absent.

    Explicit env-var overrides win:  PET_IMAGE_DIR / PET_ANNOTATION_DIR.
    """
    env_images = os.environ.get("PET_IMAGE_DIR")
    env_annotations = os.environ.get("PET_ANNOTATION_DIR")
    if env_images and env_annotations:
        return Path(env_images), Path(env_annotations)

    for root in _dataset_search_roots():
        annotation_dir = _find_annotation_dir(root)
        if annotation_dir is None:
            continue
        image_dir = _find_image_dir(root, annotation_dir)
        if image_dir is not None:
            return image_dir, annotation_dir
    return None, None


_image_dir, _annotation_dir = discover_dataset()

# Fall back to the repo-relative layout so the paths are always usable; the
# download helper in src/data_setup.py fills them in when they are missing.
IMAGE_DIR = _image_dir or (PROJECT_ROOT / "images")
ANNOTATION_DIR = _annotation_dir or (PROJECT_ROOT / "annotations" / "annotations")
TRIMAP_DIR = ANNOTATION_DIR / "trimaps"

LIST_FILE = ANNOTATION_DIR / "list.txt"
TRAINVAL_FILE = ANNOTATION_DIR / "trainval.txt"
TEST_FILE = ANNOTATION_DIR / "test.txt"

DATASET_FOUND = _image_dir is not None


def refresh_dataset_paths() -> bool:
    """Re-run discovery and update the module-level paths in place.

    Call this from a notebook right after downloading the dataset, so the
    already-imported `config` module points at the new files.
    """
    global IMAGE_DIR, ANNOTATION_DIR, TRIMAP_DIR
    global LIST_FILE, TRAINVAL_FILE, TEST_FILE, DATASET_FOUND

    image_dir, annotation_dir = discover_dataset()
    if image_dir is None:
        return False
    IMAGE_DIR = image_dir
    ANNOTATION_DIR = annotation_dir
    TRIMAP_DIR = ANNOTATION_DIR / "trimaps"
    LIST_FILE = ANNOTATION_DIR / "list.txt"
    TRAINVAL_FILE = ANNOTATION_DIR / "trainval.txt"
    TEST_FILE = ANNOTATION_DIR / "test.txt"
    DATASET_FOUND = True
    return True


# -- Hyperparameters --------------------------------------------------------
IMAGE_SIZE = _env("PET_IMAGE_SIZE", 256, int)       # resize images/masks to NxN
BATCH_SIZE = _env("PET_BATCH_SIZE", 16 if IN_NOTEBOOK_CLOUD else 8, int)
NUM_EPOCHS = _env("PET_NUM_EPOCHS", 30, int)
LEARNING_RATE = _env("PET_LEARNING_RATE", 1e-3, float)
WEIGHT_DECAY = _env("PET_WEIGHT_DECAY", 1e-5, float)

# -- Multi-task loss weighting ----------------------------------------------
# total_loss = seg_loss + CLASSIFICATION_LOSS_WEIGHT * cls_loss
CLASSIFICATION_LOSS_WEIGHT = _env("PET_CLS_LOSS_WEIGHT", 1.0, float)

# -- Segmentation inference threshold ---------------------------------------
SEGMENTATION_THRESHOLD = _env("PET_SEG_THRESHOLD", 0.5, float)

# -- Reproducibility --------------------------------------------------------
RANDOM_SEED = _env("PET_RANDOM_SEED", 42, int)
VAL_RATIO = _env("PET_VAL_RATIO", 0.1, float)  # fraction of trainval used for val

# -- DataLoader settings ----------------------------------------------------
# Worker processes need a cheap fork(); on Windows spawn makes them a liability,
# so keep 0 there and use 2 on the Linux VMs Colab / Kaggle provide.
_default_workers = 2 if (IN_NOTEBOOK_CLOUD or os.name != "nt") else 0
NUM_WORKERS = _env("PET_NUM_WORKERS", _default_workers, int)
PIN_MEMORY = torch.cuda.is_available()

# -- Device -----------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Mixed precision --------------------------------------------------------
# AMP is a CUDA-only speed-up here; disabling it on CPU avoids a crash.
USE_AMP = _env("PET_USE_AMP", True, bool) and torch.cuda.is_available()


def validate_cuda():
    """Raise an error if CUDA is not available.  Call this only inside
    training / evaluation / demo scripts, never at import time."""
    if not torch.cuda.is_available():
        if IN_COLAB:
            hint = "Colab: Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU."
        elif IN_KAGGLE:
            hint = "Kaggle: sidebar -> Session options -> Accelerator -> GPU T4 x2 / P100."
        else:
            hint = "Install a CUDA-enabled PyTorch build and run on an NVIDIA GPU."
        raise RuntimeError(
            "CUDA is required for this project, but no CUDA-enabled GPU "
            f"was detected.\n{hint}"
        )


def describe_environment() -> str:
    """One-shot summary printed at the top of notebooks / training runs."""
    if IN_COLAB:
        env_name = "Google Colab"
    elif IN_KAGGLE:
        env_name = "Kaggle"
    else:
        env_name = "local"

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"
    lines = [
        f"Environment       : {env_name}",
        f"PyTorch           : {torch.__version__}",
        f"GPU               : {gpu}",
        f"Device            : {DEVICE}",
        f"Mixed precision   : {USE_AMP}",
        f"Project root      : {PROJECT_ROOT}",
        f"Images            : {IMAGE_DIR}  (found: {IMAGE_DIR.is_dir()})",
        f"Annotations       : {ANNOTATION_DIR}  (found: {ANNOTATION_DIR.is_dir()})",
        f"Checkpoints       : {CHECKPOINT_DIR}",
        f"Results           : {RESULTS_DIR}",
        f"Image size        : {IMAGE_SIZE}",
        f"Batch size        : {BATCH_SIZE}",
        f"Epochs            : {NUM_EPOCHS}",
        f"DataLoader workers: {NUM_WORKERS}",
    ]
    return "\n".join(lines)
