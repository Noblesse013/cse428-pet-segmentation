"""
Three-class segmentation utilities (Bonus Task 3).

Instead of merging boundary pixels (trimap=3) into foreground, we keep all
three pixel categories as separate classes:

    Class 0 → Foreground  (original trimap value 1)
    Class 1 → Background  (original trimap value 2)
    Class 2 → Boundary    (original trimap value 3)

This makes segmentation a 3-class problem requiring CrossEntropyLoss and
per-class IoU / Dice metrics.
"""

import numpy as np
import torch


def process_mask_3class(mask_array: np.ndarray) -> np.ndarray:
    """
    Convert Oxford trimap to a 3-class integer mask.

    Original values: 1=foreground, 2=background, 3=boundary
    Output   values: 0=foreground, 1=background, 2=boundary   (0-indexed)

    Returns int64 array with shape (H, W).
    """
    out = np.clip(mask_array.astype(np.int64) - 1, 0, 2)
    return out


def multiclass_iou(logits: torch.Tensor, targets: torch.Tensor,
                   num_classes: int = 3, eps: float = 1e-6) -> float:
    """Mean IoU across all classes for 3-class segmentation.

    Args:
        logits:  [B, num_classes, H, W] raw model output
        targets: [B, H, W]             long integer class indices
    """
    preds = logits.argmax(dim=1)
    ious = []
    for c in range(num_classes):
        p_c = (preds == c).float()
        t_c = (targets == c).float()
        inter = (p_c * t_c).sum()
        union = p_c.sum() + t_c.sum() - inter
        ious.append(((inter + eps) / (union + eps)).item())
    return float(np.mean(ious))


def multiclass_dice(logits: torch.Tensor, targets: torch.Tensor,
                    num_classes: int = 3, eps: float = 1e-6) -> float:
    """Mean Dice coefficient across all classes for 3-class segmentation."""
    preds = logits.argmax(dim=1)
    dices = []
    for c in range(num_classes):
        p_c = (preds == c).float()
        t_c = (targets == c).float()
        inter = (p_c * t_c).sum()
        dices.append(((2.0 * inter + eps) / (p_c.sum() + t_c.sum() + eps)).item())
    return float(np.mean(dices))


def pixel_accuracy_multiclass(logits: torch.Tensor,
                               targets: torch.Tensor) -> float:
    """Fraction of correctly classified pixels."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
