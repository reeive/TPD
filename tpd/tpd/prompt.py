"""Prompt parameter flattening and updates for ViT / CLIP VPT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn


@dataclass
class PromptSnapshot:
    vector: torch.Tensor
    tensors: List[torch.Tensor]


class PromptParameterAdapter:
    """Flatten / update / restore prompt parameters for ViT or CLIP backbones."""

    def __init__(self, model: nn.Module):
        self._params = self._discover(model)
        if not self._params:
            raise ValueError("No prompt parameters found.")
        self.device = self._params[0].device
        self.shapes = [tuple(p.shape) for p in self._params]
        self.numels = [int(p.numel()) for p in self._params]
        self.dim = int(sum(self.numels))

    @staticmethod
    def _discover(model: nn.Module) -> List[nn.Parameter]:
        if hasattr(model, "prompt_learner"):
            pl = model.prompt_learner
            if hasattr(pl, "ctx") and pl.ctx is not None:
                return [pl.ctx]
        return [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and ("prompt" in n.lower() or ".ctx" in n.lower())
        ]

    def parameters(self) -> Sequence[nn.Parameter]:
        return self._params

    def vector(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self._params])

    def grad_vector(self) -> torch.Tensor:
        parts = []
        for p in self._params:
            parts.append(
                p.grad.detach().reshape(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
            )
        return torch.cat(parts)

    def apply_update(self, update: torch.Tensor) -> None:
        offset = 0
        for p, n, s in zip(self._params, self.numels, self.shapes):
            p.data.add_(update[offset : offset + n].view(s).to(p.dtype))
            offset += n

    def snapshot(self) -> PromptSnapshot:
        return PromptSnapshot(
            vector=self.vector().clone(),
            tensors=[p.detach().clone() for p in self._params],
        )

    def restore(self, snap: PromptSnapshot) -> None:
        for p, t in zip(self._params, snap.tensors):
            p.data.copy_(t.to(device=p.device, dtype=p.dtype))
