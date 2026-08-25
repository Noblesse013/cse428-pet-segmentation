"""
Interactive demo: load a trained model and visualise predictions for any image index.

Usage:
    python demo.py --model unet --index 145
    python demo.py --model attention_unet --index 145
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import config
from config import (
    IMAGE_DIR, ANNOTATION_DIR, TRIMAP_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    IMAGE_SIZE, DEVICE, RANDOM_SEED, VAL_RATIO,
)
from src.dataset import get_splits, OxfordPetDataset, process_mask
from src.transforms import get_eval_transform
from src.models.unet import BaseUNet
from src.models.attention_unet import AttentionUNet
from src.metrics import compute_iou
from src.visualization import show_prediction


def predict_by_index(
    index,
    model,
    dataset,
    device,
    idx_to_class,
    threshold=0.5,
):
    """Run inference on a single sample by its index in the dataset.

    Returns the dictionary needed for visualisation.
    """
    model.eval()
    sample = dataset[index]
    image_tensor = sample["image"].unsqueeze(0).to(device, non_blocking=True)
    true_mask = sample["mask"]       # [1, H, W]
    true_label = sample["label"]
    true_breed = sample["breed"]
    true_species = sample["species"]

    with torch.no_grad():
        seg_logits, cls_logits = model(image_tensor)

    # Segmentation prediction
    pred_prob = torch.sigmoid(seg_logits).squeeze(0).cpu()
    pred_mask = (pred_prob >= threshold).float()

    # Classification prediction
    pred_class = cls_logits.argmax(dim=1).item()
    pred_breed = idx_to_class[pred_class]

    # IoU for this single image
    iou = compute_iou(seg_logits, true_mask.unsqueeze(0).to(device), threshold)

    return {
        "image": sample["image"],
        "true_mask": true_mask,
        "pred_mask": pred_mask,
        "true_breed": true_breed,
        "pred_breed": pred_breed,
        "true_species": true_species,
        "iou": iou,
    }


def main():
    parser = argparse.ArgumentParser(description="Demo: visualise model predictions")
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "attention_unet"])
    parser.add_argument("--index", type=int, required=True,
                        help="Index of the sample to visualise")
    args = parser.parse_args()

    config.validate_cuda()
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    # ── Load splits ─────────────────────────────────────────────────────
    train_entries, val_entries, test_entries, \
        class_to_idx, idx_to_class, breed_species = get_splits(
            ANNOTATION_DIR, IMAGE_DIR, TRIMAP_DIR,
            val_ratio=VAL_RATIO, seed=RANDOM_SEED,
        )
    num_classes = len(class_to_idx)

    # ── Build combined dataset (train+val+test) for index access ────────
    all_entries = train_entries + val_entries + test_entries
    eval_tf = get_eval_transform(IMAGE_SIZE)
    full_ds = OxfordPetDataset(
        IMAGE_DIR, TRIMAP_DIR, all_entries, class_to_idx,
        image_size=IMAGE_SIZE, transform=eval_tf, mode="val",
    )

    if args.index < 0 or args.index >= len(full_ds):
        print(f"ERROR: index {args.index} out of range (0-{len(full_ds)-1})")
        sys.exit(1)

    # ── Load checkpoint ─────────────────────────────────────────────────
    ckpt_name = "best_unet.pth" if args.model == "unet" else "best_attention_unet.pth"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Train the model first.")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    if args.model == "unet":
        model = BaseUNet(in_channels=3, num_classes=num_classes, base_features=64)
    else:
        model = AttentionUNet(in_channels=3, num_classes=num_classes, base_features=64)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)

    # ── Run prediction ──────────────────────────────────────────────────
    result = predict_by_index(
        args.index, model, full_ds, DEVICE, idx_to_class,
    )

    print(f"\nOriginal class = '{result['true_breed']}' ({result['true_species']})")
    print(f"Predicted class = '{result['pred_breed']}'")
    print(f"IoU = {result['iou'] * 100:.2f}%")

    # ── Save visualisation ──────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(RESULTS_DIR / f"demo_{args.model}_{args.index}.png")
    show_prediction(
        result["image"], result["true_mask"], result["pred_mask"],
        result["true_breed"], result["pred_breed"], result["true_species"],
        result["iou"], save_path=save_path,
    )
    print(f"\nVisualisation saved to: {save_path}")


if __name__ == "__main__":
    main()
