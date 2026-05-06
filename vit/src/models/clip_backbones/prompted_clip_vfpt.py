"""VFPT (Fourier visual prompts) on CLIP ViT-B/16 — matches default vit config (fft/all)."""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn


class _FNet2D(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real


class CLIPVisualPromptVFPT(nn.Module):
    """VFPT with FOURIER_TYPE=fft, FOURIER_DIMENSION=all, FOURIER_FIRST_LAYER=True (defaults)."""

    def __init__(
        self,
        visual: nn.Module,
        num_tokens: int = 10,
        dropout: float = 0.0,
        fourier_percentage: float = 0.5,
        fourier_location: str = "prepend",
        deep: bool = True,
    ):
        super().__init__()
        object.__setattr__(self, "_v", visual)
        self.num_tokens = num_tokens
        self.deep = deep
        self.fourier_location = fourier_location
        width = int(visual.conv1.out_channels)
        patch_size = int(visual.conv1.kernel_size[0])
        layers = len(visual.transformer.resblocks)
        val = math.sqrt(6.0 / float(3 * patch_size * patch_size + width))
        self.prompt_dropout = nn.Dropout(dropout)
        self.prompt_proj = nn.Identity()
        self.FT = _FNet2D()
        self.fourier_num_tokens = max(
            1, math.floor(num_tokens * fourier_percentage)
        )
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, width))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        if deep:
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(layers - 1, num_tokens, width)
            )
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        else:
            self.register_parameter("deep_prompt_embeddings", None)

    def _patch_embed(self, x: torch.Tensor):
        v = self._v
        B = x.shape[0]
        width = int(v.conv1.out_channels)
        x = v.conv1(x)
        x = x.reshape(B, width, -1).permute(0, 2, 1)
        pe = v.positional_embedding.to(dtype=x.dtype, device=x.device)
        cls_tok = v.class_embedding.to(dtype=x.dtype, device=x.device) + pe[0:1]
        cls_tok = cls_tok.unsqueeze(0).expand(B, -1, -1)
        patch_tok = x + pe[1:].unsqueeze(0)
        return torch.cat([cls_tok, patch_tok], dim=1)

    def _incorporate_shallow(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        pe_row = self.prompt_embeddings[0]
        ft_part = self.prompt_dropout(
            self.prompt_proj(pe_row[: self.fourier_num_tokens]).expand(B, -1, -1)
        )
        ft_tok = self.FT(ft_part)
        if self.fourier_num_tokens == self.num_tokens:
            return torch.cat([x[:, :1, :], ft_tok, x[:, 1:, :]], dim=1)
        rest = self.prompt_dropout(
            self.prompt_proj(pe_row[self.fourier_num_tokens :]).expand(B, -1, -1)
        )
        if self.fourier_location == "prepend":
            return torch.cat([x[:, :1, :], ft_tok, rest, x[:, 1:, :]], dim=1)
        if self.fourier_location == "append":
            return torch.cat([x[:, :1, :], rest, ft_tok, x[:, 1:, :]], dim=1)
        # random
        idx = random.sample(range(self.num_tokens), self.fourier_num_tokens)
        tmp = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)).clone()
        for j in idx:
            tmp[:, j : j + 1, :] = self.FT(
                self.prompt_dropout(
                    self.prompt_proj(pe_row[j : j + 1]).expand(B, -1, -1)
                )
            )
        return torch.cat([x[:, :1, :], tmp, x[:, 1:, :]], dim=1)

    def _deep_inject(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """hidden: LND."""
        B = hidden.size(1)
        if self.deep_prompt_embeddings is None:
            return hidden
        pe_row = self.deep_prompt_embeddings[layer_idx]
        if self.fourier_num_tokens == 0:
            dp = self.prompt_dropout(self.prompt_proj(pe_row).expand(B, -1, -1))
            dp = dp.permute(1, 0, 2)
            return torch.cat([hidden[:1], dp, hidden[1 + self.num_tokens :]], dim=0)
        ft_part = self.prompt_dropout(
            self.prompt_proj(pe_row[: self.fourier_num_tokens]).expand(B, -1, -1)
        )
        ft_tok = self.FT(ft_part).permute(1, 0, 2)
        rest = self.prompt_dropout(
            self.prompt_proj(pe_row[self.fourier_num_tokens :]).expand(B, -1, -1)
        ).permute(1, 0, 2)
        if self.fourier_num_tokens == self.num_tokens:
            dp = ft_tok
        elif self.fourier_location == "prepend":
            dp = torch.cat([ft_tok, rest], dim=0)
        elif self.fourier_location == "append":
            dp = torch.cat([rest, ft_tok], dim=0)
        else:
            idx = random.sample(range(self.num_tokens), self.fourier_num_tokens)
            tmp = (
                self.prompt_dropout(self.prompt_proj(pe_row).expand(B, -1, -1))
                .permute(1, 0, 2)
                .clone()
            )
            for j in idx:
                cell = self.FT(
                    self.prompt_dropout(
                        self.prompt_proj(pe_row[j : j + 1]).expand(B, -1, -1)
                    )
                )
                tmp[j : j + 1] = cell.permute(1, 0, 2)
            dp = tmp
        return torch.cat([hidden[:1], dp, hidden[1 + self.num_tokens :]], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self._v
        x = self._patch_embed(x)
        x = self._incorporate_shallow(x)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i, blk in enumerate(v.transformer.resblocks):
            if self.deep and self.deep_prompt_embeddings is not None and i > 0:
                x = self._deep_inject(x, i - 1)
            x = blk(x)
        x = x.permute(1, 0, 2)
        x = v.ln_post(x[:, 0, :])
        if v.proj is not None:
            x = x @ v.proj
        return x
