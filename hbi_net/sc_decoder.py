# Copyright (c) HBI-Net authors. All rights reserved.
"""Spatial-Channel Decoder (SC-Decoder).

The SC-Decoder recalibrates the multi-scale directional change features
``{D_i}`` produced by the signed-difference fusion neck before three-class
classification. As described in the manuscript, it follows a UPer decoding
path and inserts:

* a **channel** recalibration (SE-Net style, ``M_c = sigmoid(g_c(GAP(Q)))``)
  that selects spectral-semantic channels relevant to directional bloom
  change, and
* a **spatial** recalibration (CBAM style, ``M_s = sigmoid(g_s(Q))``) that
  emphasises candidate change regions and suppresses the large stable-water
  background,

so that the recalibrated feature is ``Q_hat = Q * M_c * M_s``.

Each lateral feature and the PSP top feature are recalibrated before the
top-down FPN fusion, which is exactly where the manuscript places the
spatial-channel recalibration.
"""
import torch
import torch.nn as nn

from mmseg.models.decode_heads.uper_head import UPerHead

from opencd.registry import MODELS


class ChannelAttention(nn.Module):
    """SE-Net channel recalibration: M_c = sigmoid(g_c(GAP(Q)))."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m_c = torch.sigmoid(self.fc(self.gap(x)))
        return x * m_c


class SpatialAttention(nn.Module):
    """CBAM spatial recalibration: M_s = sigmoid(g_s(Q))."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        m_s = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * m_s


class SpatialChannelRecalibration(nn.Module):
    """Q_hat = SpatialAttn(ChannelAttn(Q))."""

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


@MODELS.register_module()
class SCDecoderHead(UPerHead):
    """UPer decoding path with spatial-channel recalibration (SC-Decoder).

    Args:
        se_reduction (int): reduction ratio of the channel recalibration.
        spatial_kernel (int): kernel size of the spatial recalibration.
        Other arguments are inherited from :class:`mmseg.UPerHead`.
    """

    def __init__(self, se_reduction: int = 16, spatial_kernel: int = 7,
                 **kwargs):
        super().__init__(**kwargs)
        # one recalibration module per lateral feature ...
        self.lateral_recal = nn.ModuleList([
            SpatialChannelRecalibration(self.channels, se_reduction,
                                        spatial_kernel)
            for _ in self.lateral_convs
        ])
        # ... and one for the PSP (top) feature.
        self.psp_recal = SpatialChannelRecalibration(
            self.channels, se_reduction, spatial_kernel)

    def _forward_feature(self, inputs):
        """Re-implements UPerHead feature fusion with SC recalibration
        applied to every lateral feature and the PSP top feature."""
        inputs = self._transform_inputs(inputs)

        # build laterals (PSP recalibrated at the top level)
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_recal(self.psp_forward(inputs)))

        # recalibrate the lateral features (all but the PSP output)
        for i in range(len(self.lateral_convs)):
            laterals[i] = self.lateral_recal[i](laterals[i])

        # top-down FPN pathway
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + torch.nn.functional.interpolate(
                laterals[i], size=prev_shape, mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = torch.nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats
