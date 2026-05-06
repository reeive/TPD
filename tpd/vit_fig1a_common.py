"""Shared helpers for ImageNet-R Fig 1(a) ViT trajectory JSON (imports from vit/)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_CODE_V1 = Path(__file__).resolve().parent.parent
_VIT_ROOT = _CODE_V1 / "vit"
if str(_VIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIT_ROOT))

from run_tta_vpt import build_dataloader, build_vpt_model  # noqa: E402


def freeze_non_prompt(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad_("prompt" in n.lower())


def confident_entropy_loss(
    logits: torch.Tensor, selection_p: float
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] <= 1:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
    ratio = min(1.0, max(selection_p, 1.0 / float(logits.shape[0])))
    keep = max(1, int(round(logits.shape[0] * ratio)))
    ent = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)
    _, idx = ent.topk(keep, largest=False)
    sub = logits[idx]
    return -(sub.softmax(dim=-1) * sub.log_softmax(dim=-1)).sum(dim=-1).mean()


def build_namespace(
    *,
    data_dir: str,
    model_root: str,
    backbone: str = "sup_vitb16_224",
    num_classes: int = 200,
    num_tokens: int = 5,
    deep_prompt: bool = True,
    batch_size: int = 1,
    cropsize: int = 224,
    workers: int = 4,
    seed: int = 42,
) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint="",
        model_root=model_root,
        backbone=backbone,
        num_tokens=num_tokens,
        deep_prompt=deep_prompt,
        init_head=True,
        prompt_init_std=0.02,
        data_dir=data_dir,
        dataset="imagenet-r",
        corruption="gaussian_noise",
        severity=5,
        num_classes=num_classes,
        cropsize=cropsize,
        batch_size=batch_size,
        workers=workers,
        num_views=1,
        seed=seed,
    )


def load_vpt_imagenet_r(
    args: SimpleNamespace, device: torch.device
) -> Tuple[nn.Module, DataLoader]:
    model = build_vpt_model(args, device)
    freeze_non_prompt(model)
    loader, _ = build_dataloader(args)
    return model, loader


def running_mean_accuracy_pct(total_correct: int, total_seen: int) -> float:
    """Same as ``run_tta_vpt.py`` / full-dataset eval: ``100 * correct / seen``."""
    if total_seen <= 0:
        return 0.0
    return 100.0 * float(total_correct) / float(total_seen)


def sliding_window_acc(correct_history: List[int], window: int) -> float:
    w = min(len(correct_history), max(1, window))
    if w == 0:
        return 0.0
    return 100.0 * sum(correct_history[-w:]) / float(w)


def zeros_spectrum_20() -> List[float]:
    return [0.0] * 20


def save_records(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1)
