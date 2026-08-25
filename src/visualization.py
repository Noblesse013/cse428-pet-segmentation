"""
Visualization utilities:
    - 3×3 random-sample grid (dataset exploration)
    - Training curves (loss / metric plots)
    - Per-image demo overlay (original | true mask | pred mask)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless training
import matplotlib.pyplot as plt
import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# 3×3 dataset exploration grid
# ─────────────────────────────────────────────────────────────────────────────

def show_random_samples(dataset, n: int = 9, save_path: str = None):
    """Randomly pick *n* samples and show them in a 3×3 grid.

    Each cell: original image with a transparent true-mask overlay, titled
    'Breed (Cat/Dog)'.
    """
    indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1) if rows * cols > 1 else [axes]

    for ax, idx in zip(axes, indices):
        sample = dataset[int(idx)]
        # image is [3,H,W] tensor normalised; revert for display
        img = sample["image"]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            # undo ImageNet normalisation
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)

        mask = sample["mask"]
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().numpy()

        ax.imshow(img)
        ax.imshow(mask, cmap="Reds", alpha=0.35)
        breed = sample["breed"]
        species = sample["species"]
        ax.set_title(f"{breed} ({species})", fontsize=10)
        ax.axis("off")

    # turn off unused axes
    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation curve plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, save_path: str):
    """Plot train vs. validation total loss."""
    epochs = range(1, len(history["train_total_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_total_loss"], "o-", label="Train Total Loss")
    ax.plot(epochs, history["val_total_loss"], "s-", label="Val Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_segmentation_curves(history: dict, save_path: str):
    """Plot IoU, Dice, Pixel Accuracy for train and validation."""
    epochs = range(1, len(history["train_iou"]) + 1)
    metrics = [
        ("iou", "IoU"),
        ("dice", "Dice"),
        ("pixel_accuracy", "Pixel Accuracy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, label) in zip(axes, metrics):
        ax.plot(epochs, history[f"train_{key}"], "o-", label="Train")
        ax.plot(epochs, history[f"val_{key}"], "s-", label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Train vs Val {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_classification_curves(history: dict, save_path: str):
    """Plot Accuracy and F1 for train and validation."""
    epochs = range(1, len(history["train_class_accuracy"]) + 1)
    metrics = [
        ("class_accuracy", "Accuracy"),
        ("class_f1", "F1"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (key, label) in zip(axes, metrics):
        tk = f"train_{key}"
        vk = f"val_{key}"
        if tk in history and vk in history:
            ax.plot(epochs, history[tk], "o-", label="Train")
            ax.plot(epochs, history[vk], "s-", label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Train vs Val {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_curves(history: dict, results_dir: str, prefix: str = ""):
    """Generate and save all three plot types."""
    tag = f"{prefix}_" if prefix else ""
    plot_loss_curves(history, f"{results_dir}/{tag}loss_curves.png")
    plot_segmentation_curves(history, f"{results_dir}/{tag}seg_curves.png")
    plot_classification_curves(history, f"{results_dir}/{tag}cls_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Single-image demo visualisation
# ─────────────────────────────────────────────────────────────────────────────

def show_prediction(
    image_tensor: torch.Tensor,
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    true_breed: str,
    pred_breed: str,
    true_species: str,
    iou: float,
    save_path: str = None,
):
    """Display original image | image + true mask | image + predicted mask.

    Title shows: Original class / Predicted class / IoU.
    """
    # Convert tensors to displayable numpy
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)

    true_m = true_mask.squeeze().cpu().numpy()
    pred_m = pred_mask.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    # Image + true mask overlay
    axes[1].imshow(img)
    axes[1].imshow(true_m, cmap="Reds", alpha=0.35)
    axes[1].set_title("Image + True Mask", fontsize=12)
    axes[1].axis("off")

    # Image + predicted mask overlay
    axes[2].imshow(img)
    axes[2].imshow(pred_m, cmap="Reds", alpha=0.35)
    axes[2].set_title("Image + Predicted Mask", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(
        f"Original class = '{true_breed}' ({true_species})\n"
        f"Predicted class = '{pred_breed}'\n"
        f"IoU = {iou * 100:.2f}%",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
