"""
Image and mask transforms for training augmentation and evaluation preprocessing.

All spatial transforms are applied identically to (image, mask) pairs so that
the segmentation ground-truth stays aligned with the image.
"""

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class Compose:
    """Chain multiple (image, mask) transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip:
    """Flip image and mask horizontally with a given probability."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask


class RandomRotation:
    """Rotate image and mask by a random angle within [-degrees, +degrees].

    Uses NEAREST for mask to preserve discrete label values.
    """

    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, image: Image.Image, mask: Image.Image):
        angle = random.uniform(-self.degrees, self.degrees)
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        return image, mask


class RandomResizedCrop:
    """Crop a random region then resize back to target size.

    Keeps image and mask spatially synchronised.
    """

    def __init__(self, size: int, scale: Tuple[float, float] = (0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, image: Image.Image, mask: Image.Image):
        w, h = image.size
        area = w * h
        target_area = random.uniform(*self.scale) * area
        aspect = random.uniform(0.75, 1.33)

        crop_w = int(round((target_area * aspect) ** 0.5))
        crop_h = int(round((target_area / aspect) ** 0.5))
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)

        image = image.crop((x, y, x + crop_w, y + crop_h))
        mask = mask.crop((x, y, x + crop_w, y + crop_h))

        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        return image, mask


class ColorJitter:
    """Randomly adjust brightness and contrast (applied to image only)."""

    def __init__(self, brightness=0.3, contrast=0.3):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if random.random() < 0.5:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image, mask


class ToTensor:
    """Convert image to [3,H,W] float tensor and mask to [1,H,W] float tensor.

    Image pixel values are normalised to [0, 1].
    Mask pixel values are binarised to {0.0, 1.0}.
    """

    def __call__(self, image: Image.Image, mask: Image.Image):
        image_np = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_np = np.array(mask, dtype=np.float32)
        if mask_np.ndim == 2:
            mask_np = mask_np[np.newaxis, ...]
        else:
            mask_np = mask_np.transpose(2, 0, 1)
        mask_np = (mask_np > 0.5).astype(np.float32)
        return image_np, mask_np


class Normalize:
    """ImageNet-style channel-wise normalisation (applied to image only)."""

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __call__(self, image: np.ndarray, mask):
        """Accepts a [3,H,W] numpy array or torch tensor for image."""
        import torch
        if isinstance(image, torch.Tensor):
            for c in range(3):
                image[c] = (image[c] - self.MEAN[c]) / self.STD[c]
        else:
            for c in range(3):
                image[c] = (image[c] - self.MEAN[c]) / self.STD[c]
        return image, mask


# ── Factory helpers ─────────────────────────────────────────────────────────

def get_train_transform(image_size: int = 256):
    """Transforms for training: augmentation + normalisation."""
    from torchvision import transforms as T

    spatial_augs = Compose([
        RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.3, contrast=0.3),
    ])

    def train_transform(image: Image.Image, mask: Image.Image):
        image, mask = spatial_augs(image, mask)
        # To tensor
        img_np, mask_np = ToTensor()(image, mask)
        # Normalise image
        img_np, mask_np = Normalize()(img_np, mask_np)
        import torch
        return torch.from_numpy(img_np), torch.from_numpy(mask_np)

    return train_transform


def get_eval_transform(image_size: int = 256):
    """Transforms for validation / test: resize + normalise, no augmentation."""

    def eval_transform(image: Image.Image, mask: Image.Image):
        image = image.resize((image_size, image_size), Image.BILINEAR)
        mask = mask.resize((image_size, image_size), Image.NEAREST)
        img_np, mask_np = ToTensor()(image, mask)
        img_np, mask_np = Normalize()(img_np, mask_np)
        import torch
        return torch.from_numpy(img_np), torch.from_numpy(mask_np)

    return eval_transform
