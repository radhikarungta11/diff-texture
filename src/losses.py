from __future__ import annotations

import torch


def masked_l1(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = torch.abs(pred_rgb - gt_rgb) * mask
    denom = mask.sum() * pred_rgb.shape[-1] + eps
    return diff.sum() / denom


def total_variation_loss(texture_map: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(texture_map[:, 1:, :, :] - texture_map[:, :-1, :, :]).mean()
    dw = torch.abs(texture_map[:, :, 1:, :] - texture_map[:, :, :-1, :]).mean()
    return dh + dw