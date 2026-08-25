"""
Base U-Net with a breed-classification head attached to the bottleneck.

Architecture:

    Encoder  (4 downsampling stages)
        ↓
    Bottleneck
        ├── Segmentation Decoder  (4 upsampling stages → 1-ch logits)
        └── Classification Head   (AdaptiveAvgPool → FC → num_classes)

Both tasks share the encoder so useful low/mid-level features learned for
segmentation also help classification and vice-versa.
"""

import torch
import torch.nn as nn

from .blocks import DoubleConv, DownBlock, UpBlock


class BaseUNet(nn.Module):
    """U-Net with encoder–decoder for segmentation + classifier on bottleneck.

    Args:
        in_channels:   input channels (3 for RGB)
        num_classes:   number of breed classes
        base_features: feature maps in the first encoder level
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 37,
                 base_features: int = 64):
        super().__init__()
        f = base_features  # shorthand

        # ── Encoder ─────────────────────────────────────────────────────
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)
        self.enc4 = DownBlock(f * 4, f * 8)

        # ── Bottleneck ──────────────────────────────────────────────────
        self.bottleneck = DownBlock(f * 8, f * 16)

        # ── Decoder (segmentation path) ─────────────────────────────────
        self.dec4 = UpBlock(f * 16 + f * 8, f * 8)
        self.dec3 = UpBlock(f * 8 + f * 4, f * 4)
        self.dec2 = UpBlock(f * 4 + f * 2, f * 2)
        self.dec1 = UpBlock(f * 2 + f, f)

        # 1×1 conv produces raw logits (no sigmoid — used with BCEWithLogitsLoss)
        self.seg_head = nn.Conv2d(f, 1, kernel_size=1)

        # ── Classification head ─────────────────────────────────────────
        # Operates on the bottleneck representation (deepest features).
        # AdaptiveAvgPool2d reduces any spatial size to 1×1, then a FC
        # layer maps to class logits.
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(f * 16, num_classes),
        )

    def forward(self, x):
        """Return (seg_logits, cls_logits).

        seg_logits: [B, 1, H, W]  — raw per-pixel logits
        cls_logits: [B, num_classes] — raw class logits
        """
        # Encoder forward — store intermediate features for skip connections
        e1 = self.enc1(x)          # [B, f,   H,   W]
        e2 = self.enc2(e1)         # [B, f*2, H/2, W/2]
        e3 = self.enc3(e2)         # [B, f*4, H/4, W/4]
        e4 = self.enc4(e3)         # [B, f*8, H/8, W/8]

        b = self.bottleneck(e4)    # [B, f*16, H/16, W/16]

        # Classification from bottleneck
        cls_logits = self.cls_head(b)

        # Decoder (skip connections restore spatial detail)
        d4 = self.dec4(b, e4)      # [B, f*8, H/8, W/8]
        d3 = self.dec3(d4, e3)     # [B, f*4, H/4, W/4]
        d2 = self.dec2(d3, e2)     # [B, f*2, H/2, W/2]
        d1 = self.dec1(d2, e1)     # [B, f,   H,   W]

        seg_logits = self.seg_head(d1)  # [B, 1, H, W]
        return seg_logits, cls_logits
