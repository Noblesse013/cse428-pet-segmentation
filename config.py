"""
Central configuration for the Oxford-IIIT Pet multi-task project.
All paths are relative to the project root so the code is portable.
"""

from pathlib import Path
import torch

# ── Paths (relative to this file) ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR = PROJECT_ROOT / "images"
ANNOTATION_DIR = PROJECT_ROOT / "annotations" / "annotations"  # nested directory
TRIMAP_DIR = ANNOTATION_DIR / "trimaps"

LIST_FILE = ANNOTATION_DIR / "list.txt"
TRAINVAL_FILE = ANNOTATION_DIR / "trainval.txt"
TEST_FILE = ANNOTATION_DIR / "test.txt"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Hyperparameters ─────────────────────────────────────────────────────────
IMAGE_SIZE = 256          # resize all images/masks to 256x256
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# ── Multi-task loss weighting ───────────────────────────────────────────────
# total_loss = seg_loss + CLASSIFICATION_LOSS_WEIGHT * cls_loss
CLASSIFICATION_LOSS_WEIGHT = 1.0

# ── Segmentation inference threshold ────────────────────────────────────────
SEGMENTATION_THRESHOLD = 0.5

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
VAL_RATIO = 0.1  # fraction of trainval used for validation

# ── DataLoader settings ─────────────────────────────────────────────────────
NUM_WORKERS = 0          # safe default for Windows / VS Code
PIN_MEMORY = True        # faster host→device transfer when using CUDA

# ── Mixed precision (optional) ──────────────────────────────────────────────
USE_AMP = True           # set to False to disable automatic mixed precision

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda")


def validate_cuda():
    """Raise an error if CUDA is not available.  Call this only inside
    training / evaluation / demo scripts, never at import time."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this project, but no CUDA-enabled GPU "
            "was detected.  Install a CUDA-enabled PyTorch build and run "
            "on an NVIDIA GPU."
        )
