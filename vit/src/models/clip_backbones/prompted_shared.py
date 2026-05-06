"""Shared helpers for CLIP visual prompting."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def expand_positional_embedding(
    positional_embedding: torch.Tensor,
    num_prompt_tokens: int,
) -> torch.Tensor:
    """Expand CLIP (1+G*G, C) positional embedding to (1+P+G*G, C).

    Prompt positions reuse the first patch-row embedding (index 1), matching
    common VPT-on-CLIP practice.
    """
    pe = positional_embedding
    cls_pe = pe[0:1]
    patch_pe = pe[1:]
    if num_prompt_tokens <= 0:
        return pe
    prompt_pe = pe[1:2].expand(num_prompt_tokens, -1)
    return torch.cat([cls_pe, prompt_pe, patch_pe], dim=0)


def vit_b16_prompt_init_std(patch_size: int = 16, width: int = 768) -> float:
    return math.sqrt(6.0 / float(3 * patch_size * patch_size + width))


def freeze_module_params(mod: nn.Module, requires_grad: bool = False) -> None:
    for p in mod.parameters():
        p.requires_grad = requires_grad
