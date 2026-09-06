"""
EfficientDet U-Net: standard U-Net encoder + BiFPN decoder + classification head.

The encoder and classification head are identical to BaseUNet.
The decoder is replaced with EfficientDetDecoder (BiFPN-based).
"""

import torch.nn as nn
from .blocks import DoubleConv, DownBlock
from .bifpn import EfficientDetDecoder


class EfficientDetUNet(nn.Module):
    """
    U-Net with EfficientDet BiFPN decoder for binary segmentation.

    Args:
        in_channels:   RGB input channels (3)
        num_classes:   number of breed classes for classification head
        base_features: feature maps in the first encoder stage
        bifpn_channels: uniform feature width inside the BiFPN block
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 37,
                 base_features: int = 64, bifpn_channels: int = 128):
        super().__init__()
        f = base_features

        # ── Encoder (identical to BaseUNet) ──────────────────────────────
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)
        self.enc4 = DownBlock(f * 4, f * 8)
        self.bottleneck = DownBlock(f * 8, f * 16)

        # ── BiFPN Decoder ─────────────────────────────────────────────────
        self.decoder = EfficientDetDecoder(
            in_channels_list=[f, f * 2, f * 4, f * 8, f * 16],
            bifpn_channels=bifpn_channels,
            out_channels=1,
        )

        # ── Classification head (identical to BaseUNet) ──────────────────
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(f * 16, num_classes),
        )

    def forward(self, x):
        """Return (seg_logits [B,1,H,W], cls_logits [B,num_classes])."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)

        cls_logits = self.cls_head(b)
        seg_logits = self.decoder(e1, e2, e3, e4, b)
        return seg_logits, cls_logits
