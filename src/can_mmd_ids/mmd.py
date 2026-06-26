from __future__ import annotations

import torch


def rbf_mmd(z_source: torch.Tensor, z_target: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(a, b) ** 2
        return torch.exp(-gamma * dist)

    k_ss = kernel(z_source, z_source).mean()
    k_tt = kernel(z_target, z_target).mean()
    k_st = kernel(z_source, z_target).mean()
    return k_ss + k_tt - 2.0 * k_st

