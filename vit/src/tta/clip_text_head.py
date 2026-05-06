"""Zero-shot CLIP text classifier head for cross-dataset evaluation."""
from __future__ import annotations

import os
import sys
from typing import List

import torch
import torch.nn as nn

_CLIP_PKG_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "third_party")
)
if _CLIP_PKG_ROOT not in sys.path:
    sys.path.insert(0, _CLIP_PKG_ROOT)


def encode_class_text_features(
    clip_model: nn.Module,
    classnames: List[str],
    templates: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Return L2-normalized text features (num_classes, embed_dim)."""
    import clip as clip_pkg

    feats = []
    with torch.no_grad():
        for c in classnames:
            texts = [t.format(c.replace("_", " ")) for t in templates]
            tokens = clip_pkg.tokenize(texts).to(device)
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.mean(dim=0))
    out = torch.stack(feats, dim=0)
    out = out / out.norm(dim=-1, keepdim=True)
    return out.float()


class CLIPZeroShotHead(nn.Module):
    """logits = scale * normalize(f_image) @ W_text^T"""

    def __init__(self, text_features: torch.Tensor, logit_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("text_features", text_features)
        self.logit_scale = logit_scale

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        f = image_features / image_features.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * (f @ self.text_features.t())
