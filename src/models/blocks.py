"""
Reusable building blocks for U-Net and Attention U-Net.

DoubleConv  – two consecutive (Conv → BN → ReLU) pairs, the standard
              encoder / decoder building unit.
DownBlock   – max-pool followed by DoubleConv (encoder downsampling).
UpBlock     – bilinear upsample + skip-connection concatenation + DoubleConv.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two successive Conv-BN-ReLU blocks.

    Used at every level of both the encoder and the decoder.
    BatchNorm after every Conv stabilises training and allows higher LR.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Encoder downsampling: max-pool → DoubleConv.

    Reduces spatial dimensions by 2× while doubling feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Decoder upsampling: upsample → concat skip → DoubleConv.

    The skip connection restores spatial detail lost during downsampling.
    We use bilinear interpolation (no learned transposed convolution) to
    avoid checkerboard artefacts.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # upsample doubles spatial dims, halves channels
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle odd-size mismatches between x and skip
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel dim
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
