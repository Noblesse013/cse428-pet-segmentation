"""
Evaluate a trained model checkpoint on train / val / test splits.

Usage:
    python evaluate.py --model unet
    python evaluate.py --model attention_unet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import config
from config import (
    IMAGE_DIR, ANNOTATION_DIR, TRIMAP_DIR, RESULTS_DIR, CHECKPOINT_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, VAL_RATIO,
    RANDOM_SEED, DEVICE,
)
from src.dataset import get_splits, create_dataloaders
from src.transforms import get_eval_transform
from src.models.unet import BaseUNet
from src.models.attention_unet import AttentionUNet
from src.evaluation import full_evaluation, print_metrics_table


def load_model(model_type: str, checkpoint_path: Path, num_classes: int):
    """Instantiate a model and load the saved weights."""
    if model_type == "unet":
        model = BaseUNet(in_channels=3, num_classes=num_classes, base_features=64)
    elif model_type == "attention_unet":
        model = AttentionUNet(in_channels=3, num_classes=num_classes, base_features=64)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "attention_unet"],
                        help="Model architecture to evaluate")
    args = parser.parse_args()

    config.validate_cuda()
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    # ── Load data splits ────────────────────────────────────────────────
    train_entries, val_entries, test_entries, \
        class_to_idx, idx_to_class, breed_species = get_splits(
            ANNOTATION_DIR, IMAGE_DIR, TRIMAP_DIR,
            val_ratio=VAL_RATIO, seed=RANDOM_SEED,
        )
    num_classes = len(class_to_idx)

    eval_tf = get_eval_transform(IMAGE_SIZE)
    _, _, test_loader = create_dataloaders(
        IMAGE_DIR, TRIMAP_DIR, ANNOTATION_DIR, class_to_idx,
        train_entries, val_entries, test_entries,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        train_transform=eval_tf, eval_transform=eval_tf,
    )
    # Rebuild train/val loaders with eval transform (no augmentation)
    from torch.utils.data import DataLoader
    from src.dataset import OxfordPetDataset
    train_ds = OxfordPetDataset(
        IMAGE_DIR, TRIMAP_DIR, train_entries, class_to_idx,
        image_size=IMAGE_SIZE, transform=eval_tf, mode="train",
    )
    val_ds = OxfordPetDataset(
        IMAGE_DIR, TRIMAP_DIR, val_entries, class_to_idx,
        image_size=IMAGE_SIZE, transform=eval_tf, mode="val",
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # ── Load checkpoint ─────────────────────────────────────────────────
    ckpt_name = "best_unet.pth" if args.model == "unet" else "best_attention_unet.pth"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Train the model first.")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model(args.model, ckpt_path, num_classes)

    # ── Evaluate all splits ─────────────────────────────────────────────
    print("\nEvaluating on training set...")
    train_metrics = full_evaluation(model, train_loader, DEVICE)
    print("Evaluating on validation set...")
    val_metrics = full_evaluation(model, val_loader, DEVICE)
    print("Evaluating on test set...")
    test_metrics = full_evaluation(model, test_loader, DEVICE)

    # ── Print table ─────────────────────────────────────────────────────
    model_display = "BASE U-NET" if args.model == "unet" else "ATTENTION U-NET"
    print_metrics_table(model_display, train_metrics, val_metrics, test_metrics)

    # ── Save metrics CSV ────────────────────────────────────────────────
    import csv
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{args.model}_final_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Train", "Validation", "Test"])
        for key in train_metrics:
            if key.startswith("_"):
                continue
            writer.writerow([
                key,
                f"{train_metrics[key]:.6f}",
                f"{val_metrics[key]:.6f}",
                f"{test_metrics[key]:.6f}",
            ])
    print(f"\nMetrics saved to {csv_path}")


if __name__ == "__main__":
    main()
