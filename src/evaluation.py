"""
Comprehensive evaluation: loads a checkpoint, runs inference over an entire
split, and prints a formatted table of all metrics.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, SEGMENTATION_THRESHOLD
from .metrics import evaluate_all_metrics


@torch.no_grad()
def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = SEGMENTATION_THRESHOLD,
) -> dict:
    """Run the model on every batch and compute all metrics at once.

    Returns a dictionary with scalar metrics (no raw tensors).
    """
    model.eval()
    all_seg_logits = []
    all_seg_targets = []
    all_cls_preds = []
    all_cls_targets = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, dtype=torch.long, non_blocking=True)

        seg_logits, cls_logits = model(images)

        all_seg_logits.append(seg_logits.cpu())
        all_seg_targets.append(masks.cpu())
        all_cls_preds.append(cls_logits.argmax(dim=1).cpu().numpy())
        all_cls_targets.append(labels.cpu().numpy())

    all_seg_logits = torch.cat(all_seg_logits, dim=0)
    all_seg_targets = torch.cat(all_seg_targets, dim=0)
    all_cls_preds = np.concatenate(all_cls_preds)
    all_cls_targets = np.concatenate(all_cls_targets)

    metrics = evaluate_all_metrics(
        all_seg_logits, all_seg_targets,
        all_cls_preds, all_cls_targets,
        threshold,
    )
    return metrics


def print_metrics_table(
    model_name: str,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
):
    """Pretty-print a comparison table for train / val / test splits."""
    header = f"{'Metric':<28}{'Train':>12}{'Validation':>14}{'Test':>12}"
    sep = "=" * 66

    seg_keys = [("IoU", "iou"), ("Dice", "dice"), ("Pixel Accuracy", "pixel_accuracy")]
    cls_keys = [
        ("Accuracy", "cls_accuracy"),
        ("Precision", "cls_precision"),
        ("Recall", "cls_recall"),
        ("F1", "cls_f1"),
    ]

    print(f"\n{sep}")
    print(f"  {model_name} RESULTS")
    print(sep)
    print(header)
    print("-" * 66)
    print("SEGMENTATION")
    for label, key in seg_keys:
        tr = train_metrics.get(key, float("nan"))
        vl = val_metrics.get(key, float("nan"))
        te = test_metrics.get(key, float("nan"))
        print(f"  {label:<26}{tr:>11.4f}{vl:>14.4f}{te:>12.4f}")
    print("-" * 66)
    print("CLASSIFICATION")
    for label, key in cls_keys:
        tr = train_metrics.get(key, float("nan"))
        vl = val_metrics.get(key, float("nan"))
        te = test_metrics.get(key, float("nan"))
        print(f"  {label:<26}{tr:>11.4f}{vl:>14.4f}{te:>12.4f}")
    print(sep)
