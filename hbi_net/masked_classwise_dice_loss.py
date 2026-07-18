# Copyright (c) HBI-Net authors. All rights reserved.
"""Masked, class-wise Dice loss used by HBI-Net.

This implementation follows the manuscript definition: Dice is computed
independently for every class, only valid pixels participate, the three class
losses are weighted equally, and the denominator uses first-order sums.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencd.registry import MODELS


@MODELS.register_module()
class MaskedClasswiseDiceLoss(nn.Module):
    """Multi-class Dice loss with an explicit spatial ignore mask.

    Args:
        eps (float): Smoothing term added to numerator and denominator.
        reduction (str): Reduction over class-wise losses. Supported values
            are ``'none'``, ``'mean'``, and ``'sum'``.
        loss_weight (float): Scalar weight applied to the reduced loss.
        ignore_index (int, optional): Label value excluded from the loss.
        loss_name (str): Name exposed to MMEngine's loss dictionary.
    """

    def __init__(self,
                 eps: float = 1e-3,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 ignore_index: Optional[int] = 255,
                 loss_name: str = 'loss_dice') -> None:
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Unsupported reduction: {reduction}')
        self.eps = float(eps)
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor=None,
                reduction_override: Optional[str] = None,
                ignore_index: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        """Compute equal-weight class-wise Dice over valid pixels."""
        del avg_factor, kwargs

        if pred.ndim != 4:
            raise ValueError('pred must have shape (N, C, H, W)')
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        if target.ndim != 3:
            raise ValueError('target must have shape (N, H, W)')
        if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
            raise ValueError('pred and target batch/spatial shapes must match')

        reduction = reduction_override or self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Unsupported reduction: {reduction}')

        effective_ignore = self.ignore_index if ignore_index is None else ignore_index
        if effective_ignore is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            valid = target != effective_ignore

        num_classes = pred.shape[1]
        invalid_label = valid & ((target < 0) | (target >= num_classes))
        if torch.any(invalid_label):
            raise ValueError('target contains a class index outside [0, C)')

        safe_target = torch.where(valid, target, torch.zeros_like(target)).long()
        one_hot = F.one_hot(safe_target, num_classes=num_classes)
        # Accumulate Dice statistics in float32 to remain stable under AMP.
        probabilities = F.softmax(pred.float(), dim=1)
        one_hot = one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)

        valid_weight = valid.unsqueeze(1).to(probabilities.dtype)
        if weight is not None:
            if weight.ndim == 4 and weight.shape[1] == 1:
                weight = weight.squeeze(1)
            if weight.shape != target.shape:
                raise ValueError('pixel weight must have shape (N, H, W)')
            valid_weight = valid_weight * weight.unsqueeze(1).to(
                probabilities.dtype)

        reduce_dims = (0, 2, 3)
        intersection = (probabilities * one_hot * valid_weight).sum(
            dim=reduce_dims)
        denominator = ((probabilities * valid_weight).sum(dim=reduce_dims)
                       + (one_hot * valid_weight).sum(dim=reduce_dims))
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        class_losses = 1 - dice

        if reduction == 'none':
            loss = class_losses
        elif reduction == 'sum':
            loss = class_losses.sum()
        else:
            loss = class_losses.mean()
        return self.loss_weight * loss

    @property
    def loss_name(self) -> str:
        return self._loss_name
