"""
Dataset class for Oxford-IIIT Pet with official trainval/test splits,
breed classification, and binary segmentation mask processing.

Trimap convention (from the original dataset):
    1 = Foreground (pet)
    2 = Background
    3 = Not classified / boundary

For binary segmentation we convert:
    0 = Background  (original 2)
    1 = Foreground  (original 1 AND original 3)
Boundary pixels (3) are treated as foreground because they belong to the pet
boundary and are more useful as positive signal than as negative.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# A few Oxford-IIIT Pet JPEGs are truncated; without this a cloud run can
# die several epochs in with 'image file is truncated'.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _parse_annotation_file(filepath: Path) -> List[Dict]:
    """Parse a list/trainval/test annotation file.

    Each non-comment line has the format:
        IMAGE_NAME  CLASS_ID  SPECIES  BREED_ID
    where IMAGE_NAME has no file extension.
    """
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            image_name = parts[0]
            class_id = int(parts[1]) - 1   # convert to 0-based
            species = int(parts[2])         # 1=Cat, 2=Dog
            breed_id = int(parts[3]) - 1    # 0-based
            entries.append({
                "image_name": image_name,
                "class_id": class_id,
                "species": species,
                "breed_id": breed_id,
            })
    return entries


def _build_class_maps(entries: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str],
                                                      Dict[str, str]]:
    """Build class-to-idx, idx-to-class, and breed-to-species mappings
    from the full annotation list."""
    # Collect unique class ids and their first-seen names
    class_names: Dict[int, str] = {}
    breed_species: Dict[str, str] = {}
    for e in entries:
        cid = e["class_id"]
        if cid not in class_names:
            class_names[cid] = e["image_name"].rsplit("_", 1)[0]
        # Capitalised first letter → Cat (per the dataset README)
        name = e["image_name"].rsplit("_", 1)[0]
        breed_species[name] = "Cat" if name[0].isupper() else "Dog"

    sorted_ids = sorted(class_names.keys())
    class_to_idx = {class_names[cid]: idx for idx, cid in enumerate(sorted_ids)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class, breed_species


def process_mask(mask_array: np.ndarray) -> np.ndarray:
    """Convert the Oxford trimap to a binary mask.

    Original values: 1=foreground, 2=background, 3=boundary.
    We keep only 0 (background) and 1 (foreground/pet).
    Boundary (3) → foreground (1) because it marks the pet outline.
    """
    binary = np.zeros_like(mask_array, dtype=np.float32)
    # 1 → foreground, 3 → foreground (boundary belongs to pet)
    binary[(mask_array == 1) | (mask_array == 3)] = 1.0
    # 2 stays 0 (background)
    return binary


class OxfordPetDataset(Dataset):
    """Oxford-IIIT Pet dataset for joint segmentation + classification.

    Args:
        image_dir:      path to images/ folder
        trimap_dir:     path to annotations/trimaps/ folder
        entries:        list of dicts from _parse_annotation_file
        class_to_idx:   mapping breed name → integer label
        image_size:     target spatial size (images are resized to square)
        transform:      callable applied to (image, mask) after loading
        mode:           'train', 'val', or 'test' (affects augmentation)
    """

    def __init__(
        self,
        image_dir: Path,
        trimap_dir: Path,
        entries: List[Dict],
        class_to_idx: Dict[str, int],
        image_size: int = 256,
        transform=None,
        mode: str = "train",
    ):
        self.image_dir = image_dir
        self.trimap_dir = trimap_dir
        self.entries = entries
        self.class_to_idx = class_to_idx
        self.image_size = image_size
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.entries[idx]
        image_name = entry["image_name"]

        # ── Load image ──────────────────────────────────────────────────
        image_path = self.image_dir / f"{image_name}.jpg"
        image = Image.open(image_path).convert("RGB")

        # ── Load trimap ─────────────────────────────────────────────────
        trimap_path = self.trimap_dir / f"{image_name}.png"
        trimap = Image.open(trimap_path)

        # ── Convert trimap to binary mask ───────────────────────────────
        mask_np = np.array(trimap, dtype=np.uint8)
        mask_np = process_mask(mask_np)  # float32, values {0, 1}

        # ── Resize both to same spatial size ────────────────────────────
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)

        # ── Apply transforms (augmentation + to-tensor) ─────────────────
        if self.transform is not None:
            image, mask_pil = self.transform(image, mask_pil)

        # ── Prepare tensors ─────────────────────────────────────────────
        # image: [3, H, W] float normalised by transform
        # mask:  [1, H, W] float in {0, 1}
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        if not isinstance(mask_pil, torch.Tensor):
            mask = torch.from_numpy(np.array(mask_pil)).unsqueeze(0).float() / 255.0
        else:
            mask = mask_pil.unsqueeze(0).float() if mask_pil.dim() == 2 else mask_pil.float()

        # ── Build label from breed name ─────────────────────────────────
        breed_name = image_name.rsplit("_", 1)[0]
        class_idx = self.class_to_idx[breed_name]
        species = "Cat" if breed_name[0].isupper() else "Dog"

        return {
            "image": image,
            "mask": mask,
            "label": class_idx,
            "breed": breed_name,
            "species": species,
            "index": idx,
        }


def get_splits(
    annotation_dir: Path,
    image_dir: Path,
    trimap_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, int],
           Dict[int, str], Dict[str, str]]:
    """Return (train_entries, val_entries, test_entries, class_to_idx,
    idx_to_class, breed_species).

    The official trainval split is partitioned reproducibly into train/val.
    The official test split is kept as-is.
    """
    trainval_entries = _parse_annotation_file(annotation_dir / "trainval.txt")
    test_entries = _parse_annotation_file(annotation_dir / "test.txt")

    # Build class maps from the full list (ensures every class is represented)
    all_entries = _parse_annotation_file(annotation_dir / "list.txt")
    class_to_idx, idx_to_class, breed_species = _build_class_maps(all_entries)

    # Reproducible shuffle for train→train+val split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(trainval_entries))
    rng.shuffle(indices)

    n_val = max(1, int(len(trainval_entries) * val_ratio))
    val_indices = set(indices[:n_val].tolist())
    train_indices = set(indices[n_val:].tolist())

    train_entries = [trainval_entries[i] for i in sorted(train_indices)]
    val_entries = [trainval_entries[i] for i in sorted(val_indices)]

    return train_entries, val_entries, test_entries, class_to_idx, idx_to_class, breed_species


def create_dataloaders(
    image_dir: Path,
    trimap_dir: Path,
    annotation_dir: Path,
    class_to_idx: Dict[str, int],
    train_entries: List[Dict],
    val_entries: List[Dict],
    test_entries: List[Dict],
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    train_transform=None,
    eval_transform=None,
):
    """Build train / val / test DataLoaders.

    Returns (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import DataLoader

    train_ds = OxfordPetDataset(
        image_dir, trimap_dir, train_entries, class_to_idx,
        image_size=image_size, transform=train_transform, mode="train",
    )
    val_ds = OxfordPetDataset(
        image_dir, trimap_dir, val_entries, class_to_idx,
        image_size=image_size, transform=eval_transform, mode="val",
    )
    test_ds = OxfordPetDataset(
        image_dir, trimap_dir, test_entries, class_to_idx,
        image_size=image_size, transform=eval_transform, mode="test",
    )

    # Respawning workers every epoch costs several seconds on Colab/Kaggle.
    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )
    return train_loader, val_loader, test_loader
