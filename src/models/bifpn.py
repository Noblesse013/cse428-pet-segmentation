"""
EfficientDet BiFPN Decoder for U-Net (Bonus Task 5).

Replaces the standard U-Net skip-connection decoder with a Bidirectional
Feature Pyramid Network (BiFPN) from EfficientDet (Tan et al., 2019).

Key innovations over standard U-Net decoder:
  1. Bidirectional flow: top-down (semantics → detail) + bottom-up (detail → semantics)
  2. Fast Normalized Weighted Fusion: learnable per-scale attention weights
     Output = sum(w_i / (sum(w_j) + eps) * Input_i)
  3. Depthwise Separable Convolutions: ~9x fewer parameters vs standard conv

Reference: https://arxiv.org/abs/1911.09070
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution:
      - Depthwise 3x3: spatial filter applied per-channel independently (groups=in_ch)
      - Pointwise 1x1: cross-channel mixing to out_channels
    Reduces computation by ~8-9x vs standard 3x3 conv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class BiFPNBlock(nn.Module):
    """
    Single BiFPN layer with Fast Normalized Weighted Feature Fusion.

    Operates on 5 feature pyramid levels (p1=largest ... p5=smallest).
    Two pathways per forward pass:
      - Top-Down: p5 → p4 → p3 → p2 → p1  (high-level semantics flow down)
      - Bottom-Up: p1 → p2 → p3 → p4 → p5  (fine spatial details flow back up)

    Each fusion node uses learnable weights normalized with ReLU + epsilon:
      fused = conv( sum_i( relu(w_i) / (sum_j relu(w_j) + eps) * input_i ) )
    """

    def __init__(self, num_channels: int = 128, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

        # Top-down fusion weights (2 inputs each: current level + upsampled deeper level)
        self.w_td4 = nn.Parameter(torch.ones(2))
        self.w_td3 = nn.Parameter(torch.ones(2))
        self.w_td2 = nn.Parameter(torch.ones(2))
        self.w_td1 = nn.Parameter(torch.ones(2))

        # Bottom-up fusion weights (3 inputs for mid-levels, 2 for deepest)
        self.w_bu2 = nn.Parameter(torch.ones(3))
        self.w_bu3 = nn.Parameter(torch.ones(3))
        self.w_bu4 = nn.Parameter(torch.ones(3))
        self.w_bu5 = nn.Parameter(torch.ones(2))

        # Depthwise separable convolutions applied after each fusion node
        self.conv_td4 = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv_td3 = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv_td2 = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv_td1 = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv_bu2 = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv_bu3 = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv_bu4 = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv_bu5 = DepthwiseSeparableConv(num_channels, num_channels)

    def _fuse(self, weights, *inputs):
        """Normalized weighted fusion: relu(w) / (sum(relu(w)) + eps)."""
        w = F.relu(weights)
        w = w / (w.sum() + self.eps)
        return sum(w[i] * inp for i, inp in enumerate(inputs))

    def forward(self, p1, p2, p3, p4, p5):
        # --- TOP-DOWN PATHWAY ---
        p4_td = self.conv_td4(self._fuse(self.w_td4, p4,
                              F.interpolate(p5, size=p4.shape[-2:], mode='nearest')))
        p3_td = self.conv_td3(self._fuse(self.w_td3, p3,
                              F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')))
        p2_td = self.conv_td2(self._fuse(self.w_td2, p2,
                              F.interpolate(p3_td, size=p2.shape[-2:], mode='nearest')))
        p1_td = self.conv_td1(self._fuse(self.w_td1, p1,
                              F.interpolate(p2_td, size=p1.shape[-2:], mode='nearest')))

        # --- BOTTOM-UP PATHWAY ---
        p2_out = self.conv_bu2(self._fuse(self.w_bu2, p2, p2_td,
                               F.max_pool2d(p1_td, 2)))
        p3_out = self.conv_bu3(self._fuse(self.w_bu3, p3, p3_td,
                               F.max_pool2d(p2_out, 2)))
        p4_out = self.conv_bu4(self._fuse(self.w_bu4, p4, p4_td,
                               F.max_pool2d(p3_out, 2)))
        p5_out = self.conv_bu5(self._fuse(self.w_bu5, p5,
                               F.max_pool2d(p4_out, 2)))

        return p1_td, p2_out, p3_out, p4_out, p5_out


class EfficientDetDecoder(nn.Module):
    """
    BiFPN-based decoder that replaces the standard U-Net skip-connection decoder.

    Steps:
      1. Project all 5 encoder feature levels to a uniform channel width (bifpn_channels)
      2. Run one BiFPN block (bidirectional weighted fusion)
      3. Upsample all levels to full resolution and sum them
      4. Final conv head produces segmentation logits [B, out_channels, H, W]

    Args:
        in_channels_list: list of channel counts [enc1, enc2, enc3, enc4, bottleneck]
                          e.g. [64, 128, 256, 512, 1024] for base_features=64
        bifpn_channels:   uniform channel width used throughout BiFPN (default 128)
        out_channels:     number of segmentation output channels (1 for binary, 3 for 3-class)
    """

    def __init__(self, in_channels_list=(64, 128, 256, 512, 1024),
                 bifpn_channels: int = 128, out_channels: int = 1):
        super().__init__()
        # 1x1 projection convolutions to normalize all encoder outputs to bifpn_channels
        self.proj = nn.ModuleList([
            nn.Conv2d(c, bifpn_channels, kernel_size=1, bias=False)
            for c in in_channels_list
        ])
        self.bifpn = BiFPNBlock(num_channels=bifpn_channels)
        # Segmentation head: 3x3 → BN → ReLU → Dropout → 1x1 → logits
        self.seg_head = nn.Sequential(
            nn.Conv2d(bifpn_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, e1, e2, e3, e4, bottleneck):
        """
        Args:
            e1 .. e4: encoder skip features (largest → smallest spatial)
            bottleneck: deepest encoder feature
        Returns:
            seg_logits: [B, out_channels, H, W]
        """
        # Project to uniform channels
        p1, p2, p3, p4, p5 = [proj(f) for proj, f in
                               zip(self.proj, [e1, e2, e3, e4, bottleneck])]
        # BiFPN bidirectional fusion
        p1_o, p2_o, p3_o, p4_o, p5_o = self.bifpn(p1, p2, p3, p4, p5)
        # Fuse all scales at full resolution (p1 size)
        H, W = p1_o.shape[-2:]
        fused = (p1_o
                 + F.interpolate(p2_o, size=(H, W), mode='bilinear', align_corners=False)
                 + F.interpolate(p3_o, size=(H, W), mode='bilinear', align_corners=False)
                 + F.interpolate(p4_o, size=(H, W), mode='bilinear', align_corners=False)
                 + F.interpolate(p5_o, size=(H, W), mode='bilinear', align_corners=False))
        return self.seg_head(fused)
