"""
Training script for Attention U-Net (segmentation + breed classification).

Run:
    python train_attention_unet.py

Produces:
    checkpoints/best_attention_unet.pth
    results/attention_unet_history.csv
    results/attention_unet_loss_curves.png
    results/attention_unet_seg_curves.png
    results/attention_unet_cls_curves.png
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import config
from src.dataset import get_splits, create_dataloaders
from src.transforms import get_train_transform, get_eval_transform
from src.models.attention_unet import AttentionUNet
from src.losses import MultiTaskLoss
from src.train_utils import train_one_epoch, evaluate_one_epoch
from src.amp_compat import get_grad_scaler
from src.visualization import plot_all_curves



def main(
    num_epochs=None,
    batch_size=None,
    learning_rate=None,
    image_size=None,
    num_workers=None,
    base_features=64,
    cls_loss_weight=None,
    image_dir=None,
    trimap_dir=None,
    annotation_dir=None,
    checkpoint_dir=None,
    results_dir=None,
):
    """Train the model and return the training-history dict.

    Every argument defaults to the value in `config`, which is itself
    overridable with PET_* environment variables.  Passing them explicitly is
    the convenient path from a notebook:

        import train_unet
        history = train_unet.main(num_epochs=10, batch_size=16)
    """
    # Resolve settings at call time, so a notebook that edits `config` after
    # import (or passes overrides here) is respected.
    image_dir = config.IMAGE_DIR if image_dir is None else Path(image_dir)
    trimap_dir = config.TRIMAP_DIR if trimap_dir is None else Path(trimap_dir)
    annotation_dir = config.ANNOTATION_DIR if annotation_dir is None else Path(annotation_dir)
    checkpoint_dir = config.CHECKPOINT_DIR if checkpoint_dir is None else Path(checkpoint_dir)
    results_dir = config.RESULTS_DIR if results_dir is None else Path(results_dir)

    IMAGE_DIR, TRIMAP_DIR, ANNOTATION_DIR = image_dir, trimap_dir, annotation_dir
    CHECKPOINT_DIR, RESULTS_DIR = checkpoint_dir, results_dir

    IMAGE_SIZE = config.IMAGE_SIZE if image_size is None else image_size
    BATCH_SIZE = config.BATCH_SIZE if batch_size is None else batch_size
    NUM_EPOCHS = config.NUM_EPOCHS if num_epochs is None else num_epochs
    LEARNING_RATE = config.LEARNING_RATE if learning_rate is None else learning_rate
    NUM_WORKERS = config.NUM_WORKERS if num_workers is None else num_workers
    CLASSIFICATION_LOSS_WEIGHT = (config.CLASSIFICATION_LOSS_WEIGHT
                                  if cls_loss_weight is None else cls_loss_weight)
    PIN_MEMORY = config.PIN_MEMORY
    VAL_RATIO = config.VAL_RATIO
    RANDOM_SEED = config.RANDOM_SEED
    USE_AMP = config.USE_AMP
    DEVICE = config.DEVICE

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    config.validate_cuda()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {DEVICE} ({gpu})")

    # ── Data ────────────────────────────────────────────────────────────
    train_entries, val_entries, test_entries, \
        class_to_idx, idx_to_class, breed_species = get_splits(
            ANNOTATION_DIR, IMAGE_DIR, TRIMAP_DIR,
            val_ratio=VAL_RATIO, seed=RANDOM_SEED,
        )

    num_classes = len(class_to_idx)
    print(f"Training samples:   {len(train_entries)}")
    print(f"Validation samples: {len(val_entries)}")
    print(f"Test samples:       {len(test_entries)}")
    print(f"Number of classes:  {num_classes}")

    train_loader, val_loader, _ = create_dataloaders(
        IMAGE_DIR, TRIMAP_DIR, ANNOTATION_DIR, class_to_idx,
        train_entries, val_entries, test_entries,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        train_transform=get_train_transform(IMAGE_SIZE),
        eval_transform=get_eval_transform(IMAGE_SIZE),
    )

    # ── Model ───────────────────────────────────────────────────────────
    model = AttentionUNet(in_channels=3, num_classes=num_classes, base_features=base_features)
    model = model.to(DEVICE)
    print(f"Attention U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimiser & loss ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-5)
    criterion = MultiTaskLoss(classification_weight=CLASSIFICATION_LOSS_WEIGHT)
    scaler = get_grad_scaler(enabled=USE_AMP)

    # ── Training loop ───────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_iou = 0.0
    history = {
        "epoch": [],
        "train_total_loss": [], "val_total_loss": [],
        "train_segmentation_loss": [], "val_segmentation_loss": [],
        "train_classification_loss": [], "val_classification_loss": [],
        "train_iou": [], "val_iou": [],
        "train_dice": [], "val_dice": [],
        "train_pixel_accuracy": [], "val_pixel_accuracy": [],
        "train_class_accuracy": [], "val_class_accuracy": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": [],
        "train_class_f1": [], "val_class_f1": [],
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 50)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            scaler=scaler,
        )
        val_metrics = evaluate_one_epoch(
            model, val_loader, criterion, DEVICE,
        )

        print(f"  Train - loss: {train_metrics['total_loss']:.4f}  "
              f"seg: {train_metrics['seg_loss']:.4f}  "
              f"cls: {train_metrics['cls_loss']:.4f}  "
              f"IoU: {train_metrics['iou']:.4f}  "
              f"Dice: {train_metrics['dice']:.4f}  "
              f"Acc: {train_metrics['cls_accuracy']:.4f}")
        print(f"  Val   - loss: {val_metrics['total_loss']:.4f}  "
              f"seg: {val_metrics['seg_loss']:.4f}  "
              f"cls: {val_metrics['cls_loss']:.4f}  "
              f"IoU: {val_metrics['iou']:.4f}  "
              f"Dice: {val_metrics['dice']:.4f}  "
              f"Acc: {val_metrics['cls_accuracy']:.4f}")

        # Record history
        history["epoch"].append(epoch)
        history["train_total_loss"].append(train_metrics["total_loss"])
        history["val_total_loss"].append(val_metrics["total_loss"])
        history["train_segmentation_loss"].append(train_metrics["seg_loss"])
        history["val_segmentation_loss"].append(val_metrics["seg_loss"])
        history["train_classification_loss"].append(train_metrics["cls_loss"])
        history["val_classification_loss"].append(val_metrics["cls_loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])
        history["train_pixel_accuracy"].append(
            train_metrics.get("pixel_accuracy", 0.0))
        history["val_pixel_accuracy"].append(val_metrics["pixel_accuracy"])
        history["train_class_accuracy"].append(train_metrics["cls_accuracy"])
        history["val_class_accuracy"].append(val_metrics["cls_accuracy"])
        history["train_precision"].append(train_metrics["cls_precision"])
        history["val_precision"].append(val_metrics["cls_precision"])
        history["train_recall"].append(train_metrics["cls_recall"])
        history["val_recall"].append(val_metrics["cls_recall"])
        history["train_class_f1"].append(train_metrics["cls_f1"])
        history["val_class_f1"].append(val_metrics["cls_f1"])

        # Save best model by validation IoU
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "config": {
                    "image_size": IMAGE_SIZE,
                    "num_classes": num_classes,
                    "base_features": base_features,
                },
            }, CHECKPOINT_DIR / "best_attention_unet.pth")
            print(f"  [best] Saved best model (val IoU = {best_val_iou:.4f})")

    # ── Save history CSV ────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "attention_unet_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history.keys()))
        writer.writeheader()
        for i in range(len(history["epoch"])):
            row = {k: history[k][i] for k in history.keys()}
            writer.writerow(row)
    print(f"\nTraining history saved to {csv_path}")

    # ── Plot training curves ────────────────────────────────────────────
    plot_all_curves(history, str(RESULTS_DIR), prefix="attention_unet")
    print(f"Training curves saved to {RESULTS_DIR}")

    return history


if __name__ == "__main__":
    main()
