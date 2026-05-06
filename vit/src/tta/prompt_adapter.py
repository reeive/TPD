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
    """Flatten, update, and restore prompt parameters for ViT/CLIP backbones."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._params = self._discover_prompt_params(model)
        if not self._params:
            raise ValueError(
                "No prompt parameters found. Expected VPT-style prompt tensors "
                "or CLIP prompt_learner.ctx."
            )

        self.device = self._params[0].device
        self.shapes = [tuple(p.shape) for p in self._params]
        self.numels = [int(p.numel()) for p in self._params]
        self.dimension = int(sum(self.numels))

    @staticmethod
    def _discover_prompt_params(model: nn.Module) -> List[nn.Parameter]:
        if hasattr(model, "prompt_learner"):
            prompt_learner = model.prompt_learner
            if hasattr(prompt_learner, "ctx") and prompt_learner.ctx is not None:
                return [prompt_learner.ctx]

        prompt_params: List[nn.Parameter] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            lower = name.lower()
            if "prompt" in lower or lower.endswith(".ctx") or ".ctx" in lower:
                prompt_params.append(param)

        return prompt_params

    def parameters(self) -> Sequence[nn.Parameter]:
        return self._params

    def vector(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self._params]).to(self.device)

    def grad_vector(self, zero_if_none: bool = True) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for param in self._params:
            if param.grad is None:
                if not zero_if_none:
                    raise RuntimeError("Prompt gradient is missing.")
                parts.append(torch.zeros_like(param).reshape(-1))
            else:
                parts.append(param.grad.detach().reshape(-1))
        return torch.cat(parts).to(self.device)

    def assign_vector(self, vector: torch.Tensor) -> None:
        vector = vector.detach().to(self.device)
        if int(vector.numel()) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vector.numel()}."
            )

        offset = 0
        for param, numel, shape in zip(self._params, self.numels, self.shapes):
            chunk = vector[offset: offset + numel].view(shape).to(param.dtype)
            param.data.copy_(chunk)
            offset += numel

    def apply_update(self, update: torch.Tensor) -> None:
        update = update.detach().to(self.device)
        if int(update.numel()) != self.dimension:
            raise ValueError(
                f"Update dimension mismatch: expected {self.dimension}, got {update.numel()}."
            )

        offset = 0
        for param, numel, shape in zip(self._params, self.numels, self.shapes):
            chunk = update[offset: offset + numel].view(shape).to(param.dtype)
            param.data.add_(chunk)
            offset += numel

    def snapshot(self) -> PromptSnapshot:
        return PromptSnapshot(
            vector=self.vector().clone(),
            tensors=[param.detach().clone() for param in self._params],
        )

    def restore(self, snapshot: PromptSnapshot) -> None:
        if snapshot.tensors and len(snapshot.tensors) == len(self._params):
            for param, tensor in zip(self._params, snapshot.tensors):
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
            return
        self.assign_vector(snapshot.vector)
