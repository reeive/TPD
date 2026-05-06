#!/usr/bin/env python3
"""
Bilinear Prompt Tuning (BPT) on Google-ViT: deep variant with per-layer
random_vectors + 1x1 conv (whitening=False, random init; see Wang et al., ICCV 2025).
"""
import math

import torch
import torch.nn as nn
from torch.nn import Conv2d, Dropout
from torch.nn.modules.utils import _pair

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer


class PromptedTransformer_BPT(Transformer):
    """
    BPT-deep: num_tokens must be a perfect square (e.g. 49 = 7*7).
    Trainable: random_vectors [L, T, C], conv1x1 (per layer C->hidden).
    """

    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        assert prompt_config.DEEP
        super(PromptedTransformer_BPT, self).__init__(config, img_size, vis)

        self.prompt_config = prompt_config
        self.vit_config = config

        img_size = _pair(img_size)
        _ = _pair(config.patches["size"])  # patch grid consistency

        self.num_tokens = int(self.prompt_config.NUM_TOKENS)
        self._prompt_grid = int(math.isqrt(self.num_tokens))
        assert self._prompt_grid * self._prompt_grid == self.num_tokens, (
            f"NUM_TOKENS must be a square, got {self.num_tokens}"
        )

        self.channels = int(self.prompt_config.BPT_CHANNELS)
        self.embed_dim = int(config.hidden_size)
        self.depth = int(config.transformer["num_layers"])

        # [L, T, C] — bilinear "burst" before 1x1
        self.random_vectors = nn.Parameter(
            torch.zeros(self.depth, self.num_tokens, self.channels)
        )
        nn.init.normal_(self.random_vectors, std=0.02)

        init_mode = str(getattr(self.prompt_config, "BPT_CONV_INIT_MODE", "kaiming"))
        if init_mode not in ("kaiming", "normal"):
            raise ValueError(f"BPT_CONV_INIT_MODE must be kaiming|normal, got {init_mode}")
        if init_mode == "kaiming":
            # Same as bpt/Models/bpt_deep.py (BPT-bilinear); bias=False
            self.conv1x1 = nn.ModuleList(
                [
                    Conv2d(
                        self.channels,
                        self.embed_dim,
                        kernel_size=1,
                        bias=False,
                    )
                    for _ in range(self.depth)
                ]
            )
            for c in self.conv1x1:
                nn.init.kaiming_normal_(c.weight, a=0, mode="fan_out")
        else:
            # Small N(0,std) weights; small prompts for ablation vs kaiming
            self.conv1x1 = nn.ModuleList(
                [
                    Conv2d(
                        self.channels,
                        self.embed_dim,
                        kernel_size=1,
                        bias=True,
                    )
                    for _ in range(self.depth)
                ]
            )
            init_std = float(getattr(self.prompt_config, "BPT_CONV_INIT_STD", 1e-3))
            for c in self.conv1x1:
                nn.init.normal_(c.weight, std=init_std)
                nn.init.zeros_(c.bias)

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

    def _creat_prompts(self, batch_size: int, device, dtype) -> list:
        out = []
        for lid in range(self.depth):
            v = self.random_vectors[lid]  # T, C
            vectors = v.unsqueeze(0).expand(batch_size, -1, -1)  # B, T, C
            vectors = vectors.transpose(1, 2).contiguous()  # B, C, T
            g = self._prompt_grid
            vectors = vectors.reshape(batch_size, self.channels, g, g)
            vectors = self.conv1x1[lid](vectors)  # B, D, g, g
            toks = (
                vectors.reshape(batch_size, self.embed_dim, g * g)
                .transpose(1, 2)
                .contiguous()
            )
            toks = toks.to(device=device, dtype=dtype)
            out.append(toks)
        return out

    def incorporate_prompt(self, x):
        B = x.shape[0]
        x = self.embeddings(x)
        dev, dt = x.device, x.dtype
        prompt_tokens = self._creat_prompts(B, dev, dt)
        p0 = self.prompt_dropout(prompt_tokens[0])
        x = torch.cat((x[:, :1, :], p0, x[:, 1:, :]), dim=1)
        return x, prompt_tokens

    def train(self, mode=True):
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            self.conv1x1.train()
        else:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output, prompt_tokens):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                p = self.prompt_dropout(prompt_tokens[i].expand(B, -1, -1))
                hidden_states = torch.cat(
                    (
                        hidden_states[:, :1, :],
                        p,
                        hidden_states[:, (1 + self.num_tokens) :, :],
                    ),
                    dim=1,
                )
                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        embedding_output, prompt_tokens = self.incorporate_prompt(x)
        encoded, attn_weights = self.forward_deep_prompt(embedding_output, prompt_tokens)
        return encoded, attn_weights


class PromptedVisionTransformer_BPT(VisionTransformer):
    def __init__(self, prompt_cfg, model_type, img_size=224, num_classes=21843, vis=False):
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer_BPT, self).__init__(model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg required for PromptedVisionTransformer_BPT")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer_BPT(
            prompt_cfg, vit_cfg, img_size, vis
        )

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        x = x[:, 0]
        logits = self.head(x)
        if not vis:
            return logits
        return logits, attn_weights

    def forward_features(self, x):
        x, attn_weights = self.transformer(x)
        cls_features = x[:, 0]
        logits = self.head(cls_features)
        return cls_features, logits, attn_weights
