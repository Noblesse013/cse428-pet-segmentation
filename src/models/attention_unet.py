"""
Attention U-Net with a breed-classification head.

Identical encoder / bottleneck / decoder to BaseUNet, but every skip
connection passes through an AttentionGate that learns to focus on
task-relevant spatial regions.

    Encoder  (4 stages)
        ↓
    Bottleneck
        ├── Segmentation Decoder  (attention-gated skip connections)
        └── Classification Head   (AdaptiveAvgPool → FC → num_classes)
"""

import torch
import torch.nn as nn

from .blocks import DoubleConv, DownBlock, UpBlock
from .attention_blocks import AttentionGate


class AttentionUNet(nn.Module):
    """Attention U-Net with classification head.

    Args:
        in_channels:   input channels (3 for RGB)
        num_classes:   number of breed classes
        base_features: feature maps in the first encoder level
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 37,
                 base_features: int = 64):
        super().__init__()
        f = base_features

        # ── Encoder (same as Base U-Net) ────────────────────────────────
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)
        self.enc4 = DownBlock(f * 4, f * 8)

        # ── Bottleneck ──────────────────────────────────────────────────
        self.bottleneck = DownBlock(f * 8, f * 16)

        # ── Attention gates on skip connections ─────────────────────────
        # Each gate receives the skip from the encoder and a gating signal
        # from the next-deeper decoder level (or bottleneck for the deepest).
        self.attn4 = AttentionGate(gate_channels=f * 16, skip_channels=f * 8)
        self.attn3 = AttentionGate(gate_channels=f * 8,  skip_channels=f * 4)
        self.attn2 = AttentionGate(gate_channels=f * 4,  skip_channels=f * 2)
        self.attn1 = AttentionGate(gate_channels=f * 2,  skip_channels=f)

        # ── Decoder ─────────────────────────────────────────────────────
        self.dec4 = UpBlock(f * 16 + f * 8, f * 8)
        self.dec3 = UpBlock(f * 8 + f * 4, f * 4)
        self.dec2 = UpBlock(f * 4 + f * 2, f * 2)
        self.dec1 = UpBlock(f * 2 + f, f)

        # ── Segmentation head (raw logits, no sigmoid) ──────────────────
        self.seg_head = nn.Conv2d(f, 1, kernel_size=1)

        # ── Classification head ─────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(f * 16, num_classes),
        )

    def forward(self, x):
        """Return (seg_logits, cls_logits)."""
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        # Classification from bottleneck
        cls_logits = self.cls_head(b)

        # Decoder with attention-gated skip connections.
        # Gating signal comes from the next-deeper decoder output (or bottleneck).
        e4_gated = self.attn4(e4, b)
        d4 = self.dec4(b, e4_gated)

        e3_gated = self.attn3(e3, d4)
        d3 = self.dec3(d4, e3_gated)

        e2_gated = self.attn2(e2, d3)
        d2 = self.dec2(d3, e2_gated)

        e1_gated = self.attn1(e1, d2)
        d1 = self.dec1(d2, e1_gated)

        seg_logits = self.seg_head(d1)
        return seg_logits, cls_logits
