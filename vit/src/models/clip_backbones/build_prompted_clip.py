"""Factory: CLIP ViT-B/16 + VPT / E2VPT / VFPT / BPT for cross-dataset TTA."""
from __future__ import annotations

import math
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .clip_loader import load_clip_vit_b16
from .prompted_clip_bpt import CLIPVisualPromptBPT
from .prompted_clip_e2vpt import CLIPVisualPromptE2VPT
from .prompted_clip_vfpt import CLIPVisualPromptVFPT
from .prompted_clip_vpt import CLIPVisualPromptVPT

MethodName = Literal["vpt", "e2vpt", "vfpt", "bpt"]


class CLIPPromptTTAModel(nn.Module):
    """CLIP image encoder with visual prompts + frozen zero-shot text prototypes."""

    def __init__(
        self,
        clip_model: nn.Module,
        visual_prompt: nn.Module,
        text_features: torch.Tensor,
    ):
        super().__init__()
        self.clip = clip_model
        self.visual_prompt = visual_prompt
        self.register_buffer("text_features", text_features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        im = self.visual_prompt(image)
        im = im / im.norm(dim=-1, keepdim=True)
        scale = self.clip.logit_scale.exp()
        return scale * (im @ self.text_features.t())


def freeze_clip_visual(clip_model: nn.Module) -> None:
    for p in clip_model.visual.parameters():
        p.requires_grad = False


def build_clipprompt_model(
    method: MethodName,
    classnames: list[str],
    templates: list[str],
    device: torch.device,
    num_tokens: int = 10,
    clip_download_root: Optional[str] = None,
    clip_checkpoint: Optional[str] = None,
    bpt_channels: int = 75,
    bpt_conv_init: str = "kaiming",
    bpt_conv_init_std: float = 1e-3,
    e2vpt_share_kv: bool = True,
    e2vpt_layer_behind: bool = True,
) -> tuple[CLIPPromptTTAModel, Any]:
    """Load CLIP, build prompted visual, encode text head. Returns (model, preprocess)."""
    from src.tta.clip_text_head import encode_class_text_features

    clip_model, preprocess = load_clip_vit_b16(
        device, clip_download_root, clip_checkpoint
    )
    clip_model = clip_model.to(device)
    freeze_clip_visual(clip_model)
    visual = clip_model.visual

    nt = num_tokens
    if method == "bpt":
        if int(math.isqrt(int(nt))) ** 2 != int(nt):
            nt = 49
        prompted = CLIPVisualPromptBPT(
            visual,
            num_tokens=nt,
            channels=bpt_channels,
            conv_init_mode=bpt_conv_init,
            conv_init_std=bpt_conv_init_std,
        ).to(device)
    elif method == "e2vpt":
        prompted = CLIPVisualPromptE2VPT(
            visual,
            num_tokens=nt,
            share_kv=e2vpt_share_kv,
            layer_behind=e2vpt_layer_behind,
        ).to(device)
    elif method == "vfpt":
        prompted = CLIPVisualPromptVFPT(visual, num_tokens=nt).to(device)
    elif method == "vpt":
        prompted = CLIPVisualPromptVPT(visual, num_tokens=nt).to(device)
    else:
        raise ValueError(method)

    text_feat = encode_class_text_features(
        clip_model, classnames, templates, device
    )
    model = CLIPPromptTTAModel(clip_model, prompted, text_feat)
    return model, preprocess


def calibrate_bpt_whiten_from_loader(
    model: CLIPPromptTTAModel,
    loader: DataLoader,
    device: torch.device,
    num_images: int,
    seed: int,
) -> None:
    """Draw first ``num_images`` samples from ``loader`` and run BPT ZCA init (in-place)."""
    vp = model.visual_prompt
    if not hasattr(vp, "init_whiten_from_images"):
        raise TypeError("visual_prompt must be CLIPVisualPromptBPT for whitening")
    chunks: list[torch.Tensor] = []
    n = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        chunks.append(images)
        n += images.size(0)
        if n >= num_images:
            break
    if not chunks:
        raise RuntimeError("empty loader for BPT whitening calibration")
    cal = torch.cat(chunks, dim=0)[:num_images]
    vp.init_whiten_from_images(cal, seed=seed)


def freeze_backbone_except_prompts(model: CLIPPromptTTAModel) -> None:
    """Ensure only prompt-related tensors train (for optimizers)."""
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        n = name.lower()
        if any(
            k in n
            for k in (
                "prompt_embeddings",
                "deep_prompt_embeddings",
                "random_vectors",
                "conv1x1",
                "kv_prompt",
                "k_prompt",
                "v_prompt",
            )
        ):
            p.requires_grad = True
