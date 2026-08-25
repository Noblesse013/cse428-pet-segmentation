"""
Attention Gate for Attention U-Net.

An attention gate learns to suppress irrelevant encoder regions and highlight
useful spatial features before they are concatenated with the decoder.

    Encoder skip feature  ──→  g (gating signal)
    Decoder feature       ──→  x (query)

The gate produces an attention map in [0, 1] that is multiplied element-wise
with the skip connection, so the decoder only sees "useful" spatial info.
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018).

    Args:
        gate_channels: channels in the gating signal (from the decoder)
        skip_channels: channels in the skip connection (from the encoder)
        inter_channels: internal intermediate channels
    """

    def __init__(self, gate_channels: int, skip_channels: int,
                 inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2

        # Linear projections to a common channel dimension
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        # Output projection → single-channel attention map
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),  # attention weights in [0, 1]
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skip: encoder feature (the "what is here" signal)
            gate: decoder feature (the "what do I need" signal)
        Returns:
            attended skip features, same shape as skip.
        """
        # Upsample gate to match skip spatial size
        g = nn.functional.interpolate(
            self.W_gate(gate), size=skip.shape[2:],
            mode="bilinear", align_corners=True,
        )
        s = self.W_skip(skip)

        # Additive attention: element-wise sum → ReLU → sigmoid → [0,1] map
        attention = self.psi(self.relu(g + s))

        # Multiply skip features by attention weights
        return skip * attention
