"""
Reusable training and validation loops.

Both functions handle:
    - moving batches to CUDA (non_blocking for speed)
    - mixed-precision via torch.amp when USE_AMP is True
    - accumulating segmentation and classification predictions

The validation loop NEVER calls optimizer.step().
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from .metrics import (
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    compute_classification_accuracy,
    compute_classification_precision,
    compute_classification_recall,
    compute_classification_f1,
)
from config import DEVICE, USE_AMP, SEGMENTATION_THRESHOLD


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    threshold: float = SEGMENTATION_THRESHOLD,
) -> dict:
    """Run one training epoch.  Returns average losses and metrics."""
    model.train()
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    n_batches = 0

    all_cls_preds = []
    all_cls_targets = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and USE_AMP:
            with autocast(device_type="cuda"):
                seg_logits, cls_logits = model(images)
                loss, seg_loss, cls_loss = criterion(
                    seg_logits, cls_logits, masks, labels
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_logits, cls_logits = model(images)
            loss, seg_loss, cls_loss = criterion(
                seg_logits, cls_logits, masks, labels
            )
            loss.backward()
            optimizer.step()

        # Accumulate scalar losses (detached to avoid graph retention)
        total_total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_cls_loss += cls_loss.item()
        total_iou += compute_iou(seg_logits, masks, threshold)
        total_dice += compute_dice(seg_logits, masks, threshold)

        # Classification predictions
        cls_pred = cls_logits.argmax(dim=1).detach().cpu().numpy()
        all_cls_preds.append(cls_pred)
        all_cls_targets.append(labels.detach().cpu().numpy())

        n_batches += 1

    all_cls_preds = np.concatenate(all_cls_preds)
    all_cls_targets = np.concatenate(all_cls_targets)

    n = max(n_batches, 1)
    return {
        "total_loss": total_total_loss / n,
        "seg_loss": total_seg_loss / n,
        "cls_loss": total_cls_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
        "cls_accuracy": compute_classification_accuracy(all_cls_targets, all_cls_preds),
        "cls_precision": compute_classification_precision(all_cls_targets, all_cls_preds),
        "cls_recall": compute_classification_recall(all_cls_targets, all_cls_preds),
        "cls_f1": compute_classification_f1(all_cls_targets, all_cls_preds),
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    threshold: float = SEGMENTATION_THRESHOLD,
) -> dict:
    """Run one validation / test epoch.  No gradient computation, no optimiser step."""
    model.eval()
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_batches = 0

    all_seg_logits = []
    all_seg_targets = []
    all_cls_preds = []
    all_cls_targets = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, dtype=torch.long, non_blocking=True)

        seg_logits, cls_logits = model(images)
        loss, seg_loss, cls_loss = criterion(
            seg_logits, cls_logits, masks, labels
        )

        total_total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_cls_loss += cls_loss.item()
        total_iou += compute_iou(seg_logits, masks, threshold)
        total_dice += compute_dice(seg_logits, masks, threshold)

        all_seg_logits.append(seg_logits.cpu())
        all_seg_targets.append(masks.cpu())
        all_cls_preds.append(cls_logits.argmax(dim=1).cpu().numpy())
        all_cls_targets.append(labels.cpu().numpy())
        n_batches += 1

    all_seg_logits = torch.cat(all_seg_logits, dim=0)
    all_seg_targets = torch.cat(all_seg_targets, dim=0)
    all_cls_preds = np.concatenate(all_cls_preds)
    all_cls_targets = np.concatenate(all_cls_targets)

    n = max(n_batches, 1)
    return {
        "total_loss": total_total_loss / n,
        "seg_loss": total_seg_loss / n,
        "cls_loss": total_cls_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
        "pixel_accuracy": compute_pixel_accuracy(
            all_seg_logits, all_seg_targets, threshold
        ),
        "cls_accuracy": compute_classification_accuracy(all_cls_targets, all_cls_preds),
        "cls_precision": compute_classification_precision(all_cls_targets, all_cls_preds),
        "cls_recall": compute_classification_recall(all_cls_targets, all_cls_preds),
        "cls_f1": compute_classification_f1(all_cls_targets, all_cls_preds),
        # keep tensors for downstream full-metric evaluation
        "_seg_logits": all_seg_logits,
        "_seg_targets": all_seg_targets,
        "_cls_preds": all_cls_preds,
        "_cls_targets": all_cls_targets,
    }
