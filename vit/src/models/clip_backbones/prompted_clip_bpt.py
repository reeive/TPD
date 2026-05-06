"""BPT (bilinear / bursty) deep visual prompts on CLIP ViT-B/16."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Conv2d


class CLIPVisualPromptBPT(nn.Module):
    """BPT-deep on CLIP visual encoder. num_tokens must be a perfect square."""

    def __init__(
        self,
        visual: nn.Module,
        num_tokens: int = 49,
        channels: int = 75,
        dropout: float = 0.0,
        conv_init_mode: str = "kaiming",
        conv_init_std: float = 1e-3,
    ):
        super().__init__()
        object.__setattr__(self, "_v", visual)
        g = int(math.isqrt(int(num_tokens)))
        if g * g != int(num_tokens):
            raise ValueError(f"BPT requires square NUM_TOKENS, got {num_tokens}")
        self.num_tokens = int(num_tokens)
        self._grid = g
        self.channels = int(channels)
        width = int(visual.conv1.out_channels)
        self.depth = len(visual.transformer.resblocks)
        self.embed_dim = width
        self.prompt_dropout = nn.Dropout(dropout)
        self._needs_whiten_calibration = conv_init_mode == "whiten"

        self.random_vectors = nn.Parameter(
            torch.zeros(self.depth, self.num_tokens, self.channels)
        )
        nn.init.normal_(self.random_vectors, std=0.02)

        if conv_init_mode not in ("kaiming", "normal", "whiten"):
            raise ValueError(conv_init_mode)
        # Placeholder init; "whiten" overwrites conv1x1 in init_whiten_from_images().
        if conv_init_mode == "whiten":
            conv_init_mode = "kaiming"
        if conv_init_mode == "kaiming":
            self.conv1x1 = nn.ModuleList(
                [
                    Conv2d(self.channels, self.embed_dim, kernel_size=1, bias=False)
                    for _ in range(self.depth)
                ]
            )
            for c in self.conv1x1:
                nn.init.kaiming_normal_(c.weight, a=0, mode="fan_out")
        else:
            self.conv1x1 = nn.ModuleList(
                [
                    Conv2d(self.channels, self.embed_dim, kernel_size=1, bias=True)
                    for _ in range(self.depth)
                ]
            )
            for c in self.conv1x1:
                nn.init.normal_(c.weight, std=float(conv_init_std))
                nn.init.zeros_(c.bias)

    @torch.no_grad()
    def _hidden_after_ln1(self, x_images: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Zero-prompt ViT path: LN1 input to attention at block ``layer_idx`` (shape S,B,D)."""
        v = self._v
        B = x_images.shape[0]
        width = int(v.conv1.out_channels)
        x = v.conv1(x_images)
        x = x.reshape(B, width, -1).permute(0, 2, 1)
        pe = v.positional_embedding.to(dtype=x.dtype, device=x.device)
        cls_pe = pe[0:1]
        patch_pe = pe[1:]
        cls_tok = v.class_embedding.to(dtype=x.dtype, device=x.device) + cls_pe
        cls_tok = cls_tok.unsqueeze(0).expand(B, -1, -1)
        patch_tok = x + patch_pe.unsqueeze(0)
        x = torch.cat([cls_tok, patch_tok], dim=1)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i in range(layer_idx):
            x = v.transformer.resblocks[i](x)
        blk = v.transformer.resblocks[layer_idx]
        return blk.ln_1(x)

    @staticmethod
    def _qk_from_mha(attn: nn.MultiheadAttention, d: int) -> tuple[torch.Tensor, torch.Tensor]:
        if attn.in_proj_weight is not None:
            w = attn.in_proj_weight
            return w[:d, :].contiguous(), w[d : 2 * d, :].contiguous()
        if attn.q_proj_weight is None or attn.k_proj_weight is None:
            raise RuntimeError("CLIP MHA has no in_proj_weight or separate q/k weights")
        return attn.q_proj_weight, attn.k_proj_weight

    @torch.no_grad()
    def init_whiten_from_images(
        self,
        images: torch.Tensor,
        jitter: float = 1e-3,
        seed: Optional[int] = None,
        max_batch: int = 32,
    ) -> None:
        """
        Per-dataset ZCA-style whitening init for each layer's 1x1 conv (BPT paper §3.2, ICCV'25).
        ``images``: [N,3,H,W] CLIP-preprocessed. ~100 images is enough (Fig.3).
        """
        if images.ndim != 4:
            raise ValueError(images.shape)
        device = images.device
        dtype = images.dtype
        v = self._v
        D = self.embed_dim
        n_img = int(images.shape[0])
        if n_img < 2:
            raise ValueError(f"need >=2 images for whitening, got {n_img}")

        gen: Optional[torch.Generator] = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))

        for lid in range(self.depth):
            x_tilde_parts: list[torch.Tensor] = []
            for s in range(0, n_img, max_batch):
                e = min(s + max_batch, n_img)
                h = self._hidden_after_ln1(images[s:e], lid)
                S, B2, d2 = h.shape
                assert d2 == D
                X = h.reshape(S * B2, D)
                blk = v.transformer.resblocks[lid]
                Wq, Wk = self._qk_from_mha(blk.attn, D)
                Wq = Wq.to(device=device, dtype=dtype)
                Wk = Wk.to(device=device, dtype=dtype)
                part = Wq @ Wk.t() @ X.t()
                x_tilde_parts.append(part)
            x_tilde = torch.cat(x_tilde_parts, dim=1)
            ncols = x_tilde.shape[1]
            sigma = (x_tilde @ x_tilde.t()) / max(float(ncols), 1.0)
            sigma = sigma + float(jitter) * torch.eye(
                D, device=device, dtype=dtype
            )
            evals, evecs = torch.linalg.eigh(sigma)
            inv_sqrt = evals.clamp(min=float(jitter)).rsqrt()
            w_mat = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.t()

            if gen is None:
                noise = torch.rand(D, D, device=device, dtype=dtype)
            else:
                noise = torch.rand(D, D, device=device, dtype=dtype, generator=gen)
            ids_shuffle = noise.argsort(dim=1)
            ids_keep = ids_shuffle[:, : self.channels]
            w_sel = torch.gather(w_mat, dim=1, index=ids_keep)
            self.conv1x1[lid].weight.data.copy_(
                w_sel.reshape(D, self.channels, 1, 1).contiguous()
            )

    def _create_prompts(self, batch_size: int, device, dtype):
        out = []
        g = self._grid
        for lid in range(self.depth):
            v = self.random_vectors[lid]
            vec = v.unsqueeze(0).expand(batch_size, -1, -1)
            vec = vec.transpose(1, 2).contiguous()
            vec = vec.reshape(batch_size, self.channels, g, g)
            vec = self.conv1x1[lid](vec)
            toks = (
                vec.reshape(batch_size, self.embed_dim, g * g)
                .transpose(1, 2)
                .contiguous()
            )
            out.append(toks.to(device=device, dtype=dtype))
        return out

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
        prompt_tokens = self._create_prompts(B, x.device, x.dtype)
        p0 = self.prompt_dropout(prompt_tokens[0])
        x = torch.cat([cls_tok, p0, patch_tok], dim=1)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i, blk in enumerate(v.transformer.resblocks):
            if i > 0:
                p = self.prompt_dropout(prompt_tokens[i])
                p = p.permute(1, 0, 2)
                x = torch.cat(
                    [x[:1], p, x[1 + self.num_tokens :]],
                    dim=0,
                )
            x = blk(x)
        x = x.permute(1, 0, 2)
        x = v.ln_post(x[:, 0, :])
        if v.proj is not None:
            x = x @ v.proj
        return x
