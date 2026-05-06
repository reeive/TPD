"""VPT-style deep visual prompts on OpenAI CLIP ViT-B/16."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CLIPVisualPromptVPT(nn.Module):
    """Deep VPT on CLIP visual encoder (frozen backbone + trainable prompts)."""

    def __init__(
        self,
        visual: nn.Module,
        num_tokens: int = 10,
        dropout: float = 0.0,
        deep: bool = True,
    ):
        super().__init__()
        object.__setattr__(self, "_v", visual)
        self.num_tokens = num_tokens
        self.deep = deep
        width = int(visual.conv1.out_channels)
        patch_size = int(visual.conv1.kernel_size[0])
        layers = len(visual.transformer.resblocks)
        val = math.sqrt(6.0 / float(3 * patch_size * patch_size + width))
        self.prompt_dropout = nn.Dropout(dropout)
        self.prompt_proj = nn.Identity()
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, width))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        if deep:
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(layers - 1, num_tokens, width)
            )
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        else:
            self.register_parameter("deep_prompt_embeddings", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self._v
        B = x.shape[0]
        width = int(v.conv1.out_channels)
        x = v.conv1(x)
        x = x.reshape(B, width, -1).permute(0, 2, 1)
        pe = v.positional_embedding.to(dtype=x.dtype, device=x.device)
        cls_pe = pe[0:1]
        patch_pe = pe[1:]
        cls_tok = v.class_embedding.to(dtype=x.dtype, device=x.device) + cls_pe
        cls_tok = cls_tok.unsqueeze(0).expand(B, -1, -1)
        patch_tok = x + patch_pe.unsqueeze(0)
        prompts = self.prompt_dropout(
            self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
        )
        x = torch.cat([cls_tok, prompts, patch_tok], dim=1)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i, blk in enumerate(v.transformer.resblocks):
            if (
                self.deep
                and self.deep_prompt_embeddings is not None
                and i > 0
            ):
                dp = self.prompt_dropout(
                    self.prompt_proj(self.deep_prompt_embeddings[i - 1])
                )
                dp = dp.unsqueeze(1).expand(-1, B, -1)
                x = torch.cat([x[:1], dp, x[1 + self.num_tokens :]], dim=0)
            x = blk(x)
        x = x.permute(1, 0, 2)
        x = v.ln_post(x[:, 0, :])
        if v.proj is not None:
            x = x @ v.proj
        return x
