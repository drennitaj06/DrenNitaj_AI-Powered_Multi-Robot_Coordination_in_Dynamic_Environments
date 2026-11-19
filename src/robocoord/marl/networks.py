"""Neural networks (algorithms §6): the shared Q-network ``356 → 256 → 256 → 5`` (ReLU).

Requires PyTorch (the ``marl`` extra).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def build_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def masked_argmax(q: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Arg-max over allowed actions; ties resolve to the lowest action index (deterministic)."""
    neg = torch.finfo(q.dtype).min
    return torch.where(masks, q, torch.full_like(q, neg)).argmax(dim=1)


def configure_torch(seed: int, deterministic: bool = True, threads: int = 1) -> None:
    """Seed torch, enable deterministic algorithms and fix the intra-op thread count (ADR-8)."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.set_num_threads(threads)
