"""
Loss functions for the multi-task (segmentation + classification) model.

Segmentation losses:
    - BCEWithLogitsLoss  (pixel-level binary cross-entropy)
    - DiceLoss           (overlap-based, handles class imbalance)

Classification loss:
    - CrossEntropyLoss   (standard multi-class)

Combined loss:
    total = seg_bce + seg_dice + CLASSIFICATION_LOSS_WEIGHT * cls_loss
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Dice = 2 * |pred ∩ gt| / (|pred| + |gt| + eps)

    We work with *probabilities* (after sigmoid) so the loss is differentiable.
    Eps prevents division-by-zero when both prediction and ground-truth are empty.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:   [B, 1, H, W] raw model output
            targets:  [B, 1, H, W] binary ground-truth
        Returns:
            Scalar dice loss (1 - dice coefficient), averaged over batch.
        """
        probs = torch.sigmoid(logits)
        # Flatten spatial dims: [B, 1, H*W]
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """Combined BCE + Dice for segmentation head.

    Using both gives the model pixel-level supervision (BCE) plus
    region-overlap supervision (Dice), which improves boundary quality.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, seg_logits: torch.Tensor,
                seg_targets: torch.Tensor) -> torch.Tensor:
        return (self.bce_weight * self.bce(seg_logits, seg_targets) +
                self.dice_weight * self.dice(seg_logits, seg_targets))


class MultiTaskLoss(nn.Module):
    """Aggregates segmentation + classification losses into a single scalar.

    total = seg_loss + classification_weight * cls_loss
    """

    def __init__(self, classification_weight: float = 1.0):
        super().__init__()
        self.seg_loss_fn = SegmentationLoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.cls_weight = classification_weight

    def forward(
        self,
        seg_logits: torch.Tensor,
        cls_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_targets: torch.Tensor,
    ):
        """
        Args:
            seg_logits:  [B, 1, H, W]
            cls_logits:  [B, num_classes]
            seg_targets: [B, 1, H, W]  float binary mask
            cls_targets: [B]           long class indices
        Returns:
            (total_loss, seg_loss, cls_loss) — all scalar tensors.
        """
        seg = self.seg_loss_fn(seg_logits, seg_targets)
        cls = self.cls_loss_fn(cls_logits, cls_targets)
        total = seg + self.cls_weight * cls
        return total, seg, cls
