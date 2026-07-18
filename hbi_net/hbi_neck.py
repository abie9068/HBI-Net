# Copyright (c) HBI-Net authors. All rights reserved.
"""Hierarchical Bitemporal Interaction (HBI) Block and signed-difference
fusion neck.

This module implements the two middle stages of HBI-Net exactly as described
in the manuscript (Section "Hierarchical Bitemporal Interaction Block"):

1. The :class:`HBIBlock` receives the paired bitemporal features
   ``F1`` and ``F2`` at one scale, builds a multi-dilation context response
   ``R`` from their channel-wise concatenation, gates it with a *signed*
   temporal-difference gate ``G = sigmoid(psi(F2 - F1))``, and applies an
   anti-symmetric residual update::

       delta = R * G
       F1' = F1 - alpha * delta
       F2' = F2 + alpha * delta

   The opposite-sign update to the two dates preserves the temporal order
   (T1 -> T2) instead of collapsing it into an unsigned magnitude.

2. The :class:`HBIFusionNeck` wires one :class:`HBIBlock` after every encoder
   stage and then forms the signed directional difference
   ``D_i = F2'_i - F1'_i`` for each scale (the "signed-difference fusion
   neck"). The list of ``D_i`` is handed to the SC-Decoder.

The neck follows the Open-CD ``SiamEncoderDecoder`` neck contract: it is called
as ``neck(feats_from, feats_to)`` where each argument is a tuple of multi-scale
feature maps produced by the shared backbone.
"""
from typing import List, Sequence

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmcv.cnn import ConvModule

from opencd.registry import MODELS


class ChannelAxialContext(BaseModule):
    """Channel mixing + horizontal/vertical depth-wise context.

    Mixes channels with a 1x1 convolution and then captures row- and
    column-direction context using a horizontal ``1xk`` and a vertical ``kx1``
    depth-wise convolution, as described for the context branch of the HBI
    Block.
    """

    def __init__(self, channels: int, kernel_size: int = 7,
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='GELU')):
        super().__init__()
        pad = kernel_size // 2
        self.channel_mix = ConvModule(
            channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = nn.Conv2d(
            channels, channels, (1, kernel_size), padding=(0, pad),
            groups=channels)
        self.v_conv = nn.Conv2d(
            channels, channels, (kernel_size, 1), padding=(pad, 0),
            groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_mix(x)
        x = x + self.h_conv(x) + self.v_conv(x)
        return x


class HBIBlock(BaseModule):
    """Hierarchical Bitemporal Interaction Block for one scale.

    Args:
        channels (int): number of channels ``C`` of each single-date feature.
        dilations (Sequence[int]): dilation rates of the grouped context
            branches. Defaults to ``(1, 2, 3, 4)``.
        alpha_init (float): initial value of the learnable residual scale
            ``alpha`` (0.1 in the manuscript).
    """

    def __init__(self,
                 channels: int,
                 dilations: Sequence[int] = (1, 2, 3, 4),
                 alpha_init: float = 0.1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.channels = channels
        self.dilations = tuple(dilations)

        # --- context branch phi ---------------------------------------
        # Grouped 3x3 dilated convs on the 2C paired representation with
        # groups=C: every group holds the (T1, T2) responses of one semantic
        # channel and maps them to a single C-channel branch output.
        self.context_branches = nn.ModuleList([
            nn.Conv2d(2 * channels, channels, kernel_size=3,
                      padding=d, dilation=d, groups=channels)
            for d in self.dilations
        ])
        # 1x1 mixing conv over the concatenated (4C) branch outputs.
        self.mix = ConvModule(
            len(self.dilations) * channels, channels, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.axial = ChannelAxialContext(
            channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # Refinement: depth-wise 3x3 + 1x1.
        self.refine = nn.Sequential(
            ConvModule(channels, channels, 3, padding=1, groups=channels,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(channels, channels, 1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # --- signed-difference gating branch psi ----------------------
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
        )

        # learnable residual scale, initialised to alpha_init.
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1),
                                              float(alpha_init)))

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):
        # paired temporal representation Z = concat(F1, F2)  -> 2C channels
        z = torch.cat([f1, f2], dim=1)
        # context response R = phi(Z)
        r = torch.cat([branch(z) for branch in self.context_branches], dim=1)
        r = self.mix(r)
        r = self.axial(r)
        r = self.refine(r)
        # signed-difference gate G = sigmoid(psi(F2 - F1))
        g = torch.sigmoid(self.gate(f2 - f1))
        # directional residual and anti-symmetric update
        delta = r * g
        f1_new = f1 - self.alpha * delta
        f2_new = f2 + self.alpha * delta
        return f1_new, f2_new


@MODELS.register_module()
class HBIFusionNeck(BaseModule):
    """Stack of HBI Blocks followed by the signed-difference fusion.

    Args:
        in_channels (Sequence[int]): per-scale channel numbers of the shared
            backbone, e.g. ``[96, 192, 384, 768]`` for Swin-T.
        dilations (Sequence[int]): dilation rates used inside every HBI Block.
        alpha_init (float): initial learnable residual scale.
    """

    def __init__(self,
                 in_channels: Sequence[int] = (96, 192, 384, 768),
                 dilations: Sequence[int] = (1, 2, 3, 4),
                 alpha_init: float = 0.1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.blocks = nn.ModuleList([
            HBIBlock(c, dilations=dilations, alpha_init=alpha_init,
                     norm_cfg=norm_cfg, act_cfg=act_cfg)
            for c in in_channels
        ])

    def forward(self,
                feats_from: Sequence[torch.Tensor],
                feats_to: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        assert len(feats_from) == len(feats_to) == len(self.blocks), \
            'number of backbone scales does not match the neck configuration'
        outs = []
        for block, f1, f2 in zip(self.blocks, feats_from, feats_to):
            f1_new, f2_new = block(f1, f2)
            # signed directional difference D_i = F2' - F1'
            outs.append(f2_new - f1_new)
        return outs
