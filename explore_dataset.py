"""
Dataset exploration script.

Run:
    python explore_dataset.py

Generates a 3×3 random sample grid saved to results/exploration_samples.png.
Also prints basic dataset statistics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    IMAGE_DIR, ANNOTATION_DIR, TRIMAP_DIR,
    IMAGE_SIZE, RANDOM_SEED, VAL_RATIO,
    RESULTS_DIR,
)
from src.dataset import get_splits, OxfordPetDataset
from src.transforms import get_eval_transform
from src.visualization import show_random_samples


def main():
    # ── Load splits ─────────────────────────────────────────────────────
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

    # ── Build a temporary dataset for visualisation ─────────────────────
    eval_tf = get_eval_transform(IMAGE_SIZE)
    # Combine trainval for broader visualisation
    all_entries = train_entries + val_entries
    full_ds = OxfordPetDataset(
        IMAGE_DIR, TRIMAP_DIR, all_entries, class_to_idx,
        image_size=IMAGE_SIZE, transform=eval_tf, mode="val",
    )

    # ── 3×3 random samples ─────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(RESULTS_DIR / "exploration_samples.png")
    show_random_samples(full_ds, n=9, save_path=save_path)
    print(f"\n3×3 sample grid saved to: {save_path}")

    # ── Species distribution ────────────────────────────────────────────
    cat_count = sum(1 for e in all_entries if breed_species.get(
        e["image_name"].rsplit("_", 1)[0], "") == "Cat")
    dog_count = len(all_entries) - cat_count
    print(f"\nSpecies split (trainval): Cat={cat_count}, Dog={dog_count}")

    # ── Breed list ──────────────────────────────────────────────────────
    print("\nBreed classes:")
    for idx, name in sorted(idx_to_class.items()):
        species = breed_species.get(name, "Unknown")
        print(f"  {idx:>2d}: {name} ({species})")


if __name__ == "__main__":
    main()
