"""E2VPT (prompt on K/V) on CLIP ViT-B/16 + deep sequence prompts."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class E2VPTCLIPResidualBlock(nn.Module):
    """Custom attention with K/V prompts; references CLIP block without re-registering it."""

    def __init__(
        self,
        orig_block: nn.Module,
        num_heads: int,
        num_prompt_kv: int,
        share_kv: bool,
        layer_behind: bool,
    ):
        super().__init__()
        object.__setattr__(self, "_orig", orig_block)
        attn = orig_block.attn
        self.num_heads = num_heads
        embed_dim = attn.embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_prompt_kv = num_prompt_kv
        self.share_kv = share_kv
        self.layer_behind = layer_behind
        H, P, Dh = num_heads, num_prompt_kv, self.head_dim
        if share_kv:
            self.kv_prompt = nn.Parameter(torch.zeros(H, P, Dh))
            nn.init.uniform_(self.kv_prompt, -0.02, 0.02)
        else:
            self.k_prompt = nn.Parameter(torch.zeros(H, P, Dh))
            self.v_prompt = nn.Parameter(torch.zeros(H, P, Dh))
            nn.init.uniform_(self.k_prompt, -0.02, 0.02)
            nn.init.uniform_(self.v_prompt, -0.02, 0.02)

        d = float(getattr(attn, "dropout", 0.0))
        self._dropout = nn.Dropout(d) if d and d > 0 else nn.Identity()

    def _self_attn(self, x_ln: torch.Tensor) -> torch.Tensor:
        """x_ln: (L, B, C) LND."""
        attn_mod = self._orig.attn
        L, B, C = x_ln.shape
        H, Dh = self.num_heads, self.head_dim
        w = attn_mod.in_proj_weight
        b = attn_mod.in_proj_bias
        wq, wk, wv = w.chunk(3, dim=0)
        if b is None:
            bq = bk = bv = None
        else:
            bq, bk, bv = b.chunk(3, dim=0)
        q = F.linear(x_ln, wq, bq)
        k = F.linear(x_ln, wk, bk)
        v = F.linear(x_ln, wv, bv)
        q = q.reshape(L, B, H, Dh).permute(1, 2, 0, 3)
        k = k.reshape(L, B, H, Dh).permute(1, 2, 0, 3)
        v = v.reshape(L, B, H, Dh).permute(1, 2, 0, 3)
        if self.share_kv:
            kp = self.kv_prompt.unsqueeze(0).expand(B, -1, -1, -1)
            vp = kp
        else:
            kp = self.k_prompt.unsqueeze(0).expand(B, -1, -1, -1)
            vp = self.v_prompt.unsqueeze(0).expand(B, -1, -1, -1)
        if self.layer_behind:
            k = torch.cat([kp, k], dim=2)
            v = torch.cat([vp, v], dim=2)
        else:
            k = torch.cat([k, kp], dim=2)
            v = torch.cat([v, vp], dim=2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (Dh ** -0.5)
        prob = scores.softmax(dim=-1)
        prob = self._dropout(prob)
        out = torch.matmul(prob, v)
        out = out.permute(2, 0, 1, 3).reshape(L, B, C)
        return F.linear(out, attn_mod.out_proj.weight, attn_mod.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self._orig
        x = x + self._self_attn(o.ln_1(x))
        x = x + o.mlp(o.ln_2(x))
        return x


class CLIPVisualPromptE2VPT(nn.Module):
    """E2VPT: deep sequence prompts (VPT) + per-layer K/V prompts (default cfg)."""

    def __init__(
        self,
        visual: nn.Module,
        num_tokens: int = 10,
        num_prompt_kv: Optional[int] = None,
        dropout: float = 0.0,
        share_kv: bool = True,
        layer_behind: bool = True,
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

        heads = width // 64
        nv = num_prompt_kv if num_prompt_kv is not None else num_tokens
        self.blocks = nn.ModuleList(
            [
                E2VPTCLIPResidualBlock(
                    visual.transformer.resblocks[i],
                    heads,
                    nv,
                    share_kv,
                    layer_behind,
                )
                for i in range(layers)
            ]
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self._v
        B = x.shape[0]
        x = self._patch_embed(x)
        prompts = self.prompt_dropout(
            self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
        )
        x = torch.cat([x[:, :1, :], prompts, x[:, 1:, :]], dim=1)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i, blk in enumerate(self.blocks):
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
