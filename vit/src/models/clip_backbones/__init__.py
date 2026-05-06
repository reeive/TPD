"""CLIP ViT-B/16 + visual prompt variants for cross-dataset TTA."""

from .build_prompted_clip import build_clipprompt_model, freeze_backbone_except_prompts

__all__ = ["build_clipprompt_model", "freeze_backbone_except_prompts"]
