"""
Classifier head variants for Bonus Task 1.

Three architectures attached to the U-Net bottleneck (1024-ch feature map):
  1. MobileNetClassifier  — Depthwise Separable Convolutions (parameter-efficient)
  2. DenseNetClassifier   — Dense feature reuse blocks (feature concatenation)
  3. ConvClassifier       — Standard conv baseline (same as original cls_head reimplemented standalone)

All return raw logits [B, num_classes] for CrossEntropyLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvClassifier(nn.Module):
    """Standard baseline: AdaptiveAvgPool → Dropout → Linear."""

    def __init__(self, in_channels: int = 1024, num_classes: int = 37):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class MobileNetClassifier(nn.Module):
    """
    MobileNet-style classifier using Depthwise Separable Convolutions.

    Depthwise separable conv = depthwise 3x3 (per-channel spatial filter)
                             + pointwise 1x1 (cross-channel mixing)
    Reduces parameters by ~8-9x vs standard convolution.
    """

    def __init__(self, in_channels: int = 1024, num_classes: int = 37):
        super().__init__()
        # Depthwise-Separable Block 1: 1024 → 512
        self.dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pw1 = nn.Conv2d(in_channels, 512, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        # Depthwise-Separable Block 2: 512 → 256
        self.dw2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512, bias=False)
        self.pw2 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.drop = nn.Dropout(0.3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.pw1(self.dw1(x))))
        x = self.drop(x)
        x = F.relu(self.bn2(self.pw2(self.dw2(x))))
        x = self.drop(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


class _DenseBlock(nn.Module):
    """Single dense layer: BN → ReLU → Conv → concatenate with input."""

    def __init__(self, in_channels: int, growth_rate: int = 32):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return torch.cat([x, self.conv(F.relu(self.bn(x)))], dim=1)


class DenseNetClassifier(nn.Module):
    """
    DenseNet-style classifier using dense feature concatenation blocks.

    Each block concatenates its output with its input, so all preceding
    feature maps are reused at every subsequent layer.
    """

    def __init__(self, in_channels: int = 1024, num_classes: int = 37, growth_rate: int = 32):
        super().__init__()
        self.db1 = _DenseBlock(in_channels, growth_rate)
        self.db2 = _DenseBlock(in_channels + growth_rate, growth_rate)
        self.db3 = _DenseBlock(in_channels + 2 * growth_rate, growth_rate)
        total_ch = in_channels + 3 * growth_rate
        self.drop = nn.Dropout(0.3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(total_ch, num_classes)

    def forward(self, x):
        x = self.db1(x)
        x = self.db2(x)
        x = self.db3(x)
        x = self.drop(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)
