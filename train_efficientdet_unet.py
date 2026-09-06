"""
Training script for EfficientDet U-Net (Bonus Task 5).

Usage:
    python train_efficientdet_unet.py
"""

import sys, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import config
from src.dataset import get_splits, create_dataloaders
from src.transforms import get_train_transform, get_eval_transform
from src.models.efficientdet_unet import EfficientDetUNet
from src.losses import MultiTaskLoss
from src.train_utils import train_one_epoch, evaluate_one_epoch
from src.amp_compat import get_grad_scaler
from src.visualization import plot_all_curves


def main(
    num_epochs=None, batch_size=None, learning_rate=None,
    image_size=None, num_workers=None, base_features=64,
    bifpn_channels=128, cls_loss_weight=None,
    checkpoint_dir=None, results_dir=None,
    use_augmentation=True, weight_decay=1e-4,
):
    image_dir      = config.IMAGE_DIR
    trimap_dir     = config.TRIMAP_DIR
    annotation_dir = config.ANNOTATION_DIR
    checkpoint_dir = config.CHECKPOINT_DIR if checkpoint_dir is None else Path(checkpoint_dir)
    results_dir    = config.RESULTS_DIR    if results_dir    is None else Path(results_dir)

    IMAGE_SIZE  = config.IMAGE_SIZE  if image_size    is None else image_size
    BATCH_SIZE  = config.BATCH_SIZE  if batch_size    is None else batch_size
    NUM_EPOCHS  = config.NUM_EPOCHS  if num_epochs    is None else num_epochs
    LR          = config.LEARNING_RATE if learning_rate is None else learning_rate
    NUM_WORKERS = config.NUM_WORKERS if num_workers   is None else num_workers
    CLS_W       = config.CLASSIFICATION_LOSS_WEIGHT if cls_loss_weight is None else cls_loss_weight
    DEVICE      = config.DEVICE

    torch.manual_seed(config.RANDOM_SEED)

    config.validate_cuda()

    train_e, val_e, test_e, c2i, i2c, _ = get_splits(
        annotation_dir, image_dir, trimap_dir,
        val_ratio=config.VAL_RATIO, seed=config.RANDOM_SEED,
    )
    num_classes = len(c2i)

    train_tf = get_train_transform(IMAGE_SIZE) if use_augmentation else get_eval_transform(IMAGE_SIZE)
    train_loader, val_loader, _ = create_dataloaders(
        image_dir, trimap_dir, annotation_dir, c2i,
        train_e, val_e, test_e,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        train_transform=train_tf,
        eval_transform=get_eval_transform(IMAGE_SIZE),
    )

    model = EfficientDetUNet(in_channels=3, num_classes=num_classes,
                             base_features=base_features, bifpn_channels=bifpn_channels)
    model = model.to(DEVICE)
    print(f"EfficientDet U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    criterion = MultiTaskLoss(classification_weight=CLS_W)
    scaler    = get_grad_scaler(enabled=config.USE_AMP)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

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
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler=scaler)
        val_m   = evaluate_one_epoch(model, val_loader, criterion, DEVICE)

        print(f"  Train IoU:{train_m['iou']:.4f}  Acc:{train_m['cls_accuracy']:.4f}")
        print(f"  Val   IoU:{val_m['iou']:.4f}    Acc:{val_m['cls_accuracy']:.4f}")

        history["epoch"].append(epoch)
        for k in ["total_loss","segmentation_loss","classification_loss",
                  "iou","dice","pixel_accuracy","class_accuracy",
                  "precision","recall","class_f1"]:
            tk = k.replace("segmentation_loss","seg_loss").replace("classification_loss","cls_loss")\
                  .replace("class_accuracy","cls_accuracy").replace("class_f1","cls_f1")\
                  .replace("pixel_accuracy","pixel_accuracy")
            history[f"train_{k}"].append(train_m.get(tk, train_m.get(k, 0.0)))
            history[f"val_{k}"].append(val_m.get(tk, val_m.get(k, 0.0)))

        if val_m["iou"] > best_val_iou:
            best_val_iou = val_m["iou"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": c2i, "idx_to_class": i2c,
                "config": {"image_size": IMAGE_SIZE, "num_classes": num_classes,
                           "base_features": base_features, "bifpn_channels": bifpn_channels},
            }, checkpoint_dir / "best_efficientdet_unet.pth")
            print(f"  [best] val IoU = {best_val_iou:.4f}")

    csv_path = results_dir / "efficientdet_unet_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history.keys()))
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({k: history[k][i] for k in history})
    print(f"History → {csv_path}")

    plot_all_curves(history, str(results_dir), prefix="efficientdet_unet")
    return history


if __name__ == "__main__":
    main()
