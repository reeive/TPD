"""Load OpenAI CLIP (ViT-B/16) using vendored package under vit/src/third_party/clip."""
from __future__ import annotations

import os
import sys
from typing import Any, Optional, Tuple

import torch

_THIRD_PARTY = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party")
)
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)


def load_clip_vit_b16(
    device: torch.device,
    download_root: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    import clip as clip_pkg

    def _unpack_load(*load_args: Any, **load_kw: Any) -> Tuple[Any, Any]:
        out = clip_pkg.load(*load_args, **load_kw)
        if len(out) == 3:
            model, _embed_dim, preprocess = out
            return model, preprocess
        model, preprocess = out  # type: ignore[misc]
        return model, preprocess

    ckpt = checkpoint_path or os.environ.get("CLIP_CHECKPOINT", "").strip() or None
    if ckpt and os.path.isfile(ckpt):
        model, preprocess = _unpack_load(ckpt, device=device, jit=False)
    else:
        root = download_root or os.path.expanduser("~/.cache/clip")
        model, preprocess = _unpack_load(
            "ViT-B/16", device=device, jit=False, download_root=root
        )
    model = model.float()
    return model, preprocess
