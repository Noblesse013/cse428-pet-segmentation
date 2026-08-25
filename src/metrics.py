"""
Evaluation metrics for both tasks.

Segmentation:
    - IoU (Intersection over Union) / mIoU
    - Dice coefficient
    - Pixel accuracy

Classification:
    - Accuracy
    - Precision (macro)
    - Recall (macro)
    - F1 score (macro)

All functions are safe against division-by-zero.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(preds: torch.Tensor, targets: torch.Tensor,
                threshold: float = 0.5) -> float:
    """IoU for a batch of binary segmentation masks.

    Args:
        preds:    [B, 1, H, W] logits
        targets:  [B, 1, H, W] binary ground-truth
    Returns:
        Mean IoU across the batch.
    """
    probs = torch.sigmoid(preds)
    pred_bin = (probs >= threshold).float()
    intersection = (pred_bin * targets).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean().item()


def compute_dice(preds: torch.Tensor, targets: torch.Tensor,
                 threshold: float = 0.5) -> float:
    """Dice coefficient for a batch of binary segmentation masks."""
    probs = torch.sigmoid(preds)
    pred_bin = (probs >= threshold).float()
    intersection = (pred_bin * targets).sum(dim=(1, 2, 3))
    cardinality = pred_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1e-7) / (cardinality + 1e-7)
    return dice.mean().item()


def compute_pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor,
                           threshold: float = 0.5) -> float:
    """Pixel-level accuracy for a batch of binary segmentation masks."""
    probs = torch.sigmoid(preds)
    pred_bin = (probs >= threshold).float()
    correct = (pred_bin == targets).float().sum()
    total = targets.numel()
    return (correct / (total + 1e-7)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics  (all use macro averaging for multiclass)
# ─────────────────────────────────────────────────────────────────────────────

def compute_classification_accuracy(y_true: np.ndarray,
                                    y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def compute_classification_precision(y_true: np.ndarray,
                                     y_pred: np.ndarray) -> float:
    return precision_score(y_true, y_pred, average="macro", zero_division=0)


def compute_classification_recall(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> float:
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


def compute_classification_f1(y_true: np.ndarray,
                              y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: evaluate a full epoch and return all metrics at once
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_metrics(
    all_seg_logits: torch.Tensor,
    all_seg_targets: torch.Tensor,
    all_cls_preds: np.ndarray,
    all_cls_targets: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute every metric and return as a flat dictionary."""
    return {
        "iou": compute_iou(all_seg_logits, all_seg_targets, threshold),
        "dice": compute_dice(all_seg_logits, all_seg_targets, threshold),
        "pixel_accuracy": compute_pixel_accuracy(
            all_seg_logits, all_seg_targets, threshold
        ),
        "cls_accuracy": compute_classification_accuracy(
            all_cls_targets, all_cls_preds
        ),
        "cls_precision": compute_classification_precision(
            all_cls_targets, all_cls_preds
        ),
        "cls_recall": compute_classification_recall(
            all_cls_targets, all_cls_preds
        ),
        "cls_f1": compute_classification_f1(
            all_cls_targets, all_cls_preds
        ),
    }
