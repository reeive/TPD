#!/usr/bin/env python3
"""Shared helpers for TTA runners: synset head init, ImageFolder loader."""

import glob
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

SYNSET_SEARCH_BASES = [
    "./data",
    "./checkpoints",
]

# NOTE: ViT-B_16-224.npz from some sources may not match standard ILSVRC-1k ordering.
# Use a proper 21k+1k fine-tuned checkpoint (e.g. timm_vitb16_21kft1k) for exact 1k alignment.
BACKBONE_REGISTRY = {
    "sup_vitb16_224": ("ViT-B_16-224.npz", "1k"),
    "sup_vitb16": ("ViT-B_16.npz", "1k"),
    "sup_vitl16_224": ("ViT-L_16-224.npz", "1k"),
    "sup_vitl16": ("ViT-L_16.npz", "1k"),
    "sup_vitb16_imagenet21k": ("imagenet21k_ViT-B_16.npz", "21k"),
    "sup_vitl16_imagenet21k": ("imagenet21k_ViT-L_16.npz", "21k"),
    "sup_vith14_imagenet21k": ("imagenet21k_ViT-H_14.npz", "21k"),
    "timm_vitb16_21kft1k": ("ViT-B_16_timm_21kft1k.npz", "1k"),
}

_VIT_NORM = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def get_vit_norm_params(_backbone: str = ""):
    """Google ViT checkpoints use 0.5/0.5 normalization."""
    return _VIT_NORM


def _bundled_info_dir() -> str:
    # vit/src/tta/_common.py -> vit/data/_info
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "_info")
    )


def find_synset_file(name: str) -> Optional[str]:
    """Find a synset txt file by name across known locations."""
    bundled = os.path.join(_bundled_info_dir(), name)
    if os.path.isfile(bundled):
        return bundled
    try:
        import timm

        info_dir = os.path.join(os.path.dirname(timm.__file__), "data", "_info")
        p = os.path.join(info_dir, name)
        if os.path.isfile(p):
            return p
    except Exception:
        pass
    for base in SYNSET_SEARCH_BASES:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            if name in files:
                return os.path.join(root, name)
    hf_dirs = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--timm--*/refs")
    )
    for c in hf_dirs:
        parent = os.path.dirname(c)
        for root, _dirs, files in os.walk(parent):
            if name in files:
                return os.path.join(root, name)
    return None


def build_head_indices(
    data_dir: str, backbone: str = "sup_vitb16_224"
) -> Tuple[List[int], int]:
    """Map dataset class directories (wnids *or* integer ids) to head row
    indices in .npz.

    ImageNet-V2 的 class 目录是整数 0..999，直接对应 1k head 行；
    其他 datasets 目录是 wnid，需要经 synset 表映射。
    """
    class_dirs = sorted(
        [
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
    )
    print(f"  Found {len(class_dirs)} class directories in {data_dir}")

    # ImageFolder 按字典序排序 class dirs；若全是整数字符串，也用字典序
    # 作为 label 索引。我们把每个 dir 名解析为 int，直接作为 1k head 行索引。
    if class_dirs and all(c.isdigit() for c in class_dirs):
        indices = [int(c) for c in class_dirs]
        print(
            f"  Integer-indexed dataset: mapping dir name -> 1k head row "
            f"directly ({len(indices)} classes)."
        )
        return indices, len(indices)

    _, synset_mode = BACKBONE_REGISTRY.get(backbone, (None, "1k"))

    if synset_mode == "21k":
        synset_file = find_synset_file("imagenet21k_goog_synsets.txt")
        if synset_file is None:
            raise RuntimeError(
                "21k backbone requires imagenet21k_goog_synsets.txt. "
                "Copy from timm/data/_info/ or download from Google."
            )
        with open(synset_file) as f:
            synsets = [line.strip() for line in f if line.strip()]
        label = "ImageNet-21k"
    else:
        synset_file = find_synset_file("imagenet_synsets.txt")
        if synset_file is None:
            raise RuntimeError(
                "1k backbone requires imagenet_synsets.txt from timm."
            )
        with open(synset_file) as f:
            synsets = [line.strip().split()[0] for line in f if line.strip()]
        label = "ImageNet-1k"

    synset_to_idx = {s: i for i, s in enumerate(synsets)}

    indices = []
    missing = []
    for d in class_dirs:
        if d in synset_to_idx:
            indices.append(synset_to_idx[d])
        else:
            # 容错：若个别 wnid 不在 21k goog synset 列表里（已知 n04399382 不在），
            # 占位用索引 -1，init_head_from_npz 会将对应行置零。
            indices.append(-1)
            missing.append(d)

    if missing:
        print(
            f"  [warn] {len(missing)}/{len(class_dirs)} synsets NOT in {label} "
            f"({missing[:5]}{'...' if len(missing) > 5 else ''}); "
            "对应 head 行将置零。"
        )
    print(
        f"  {len(indices)-len(missing)}/{len(indices)} synsets mapped to "
        f"{label} indices (from {len(synsets)} total classes)."
    )
    return indices, len(indices)


def init_head_from_npz(
    model: nn.Module,
    npz_path: str,
    class_indices: List[int],
    device: torch.device,
    prompt_init_std: float = 0.02,
) -> None:
    """Initialize classification head from .npz; re-init prompt *embedding* tensors."""
    d = np.load(npz_path)

    head_w_full = torch.tensor(d["head/kernel"].T, dtype=torch.float32)
    head_b_full = torch.tensor(d["head/bias"], dtype=torch.float32)
    hidden_dim = head_w_full.shape[1]

    # 仅当 pre_logits 是训练过的（bias 非零或 kernel 非恒等）才串联 pre_logits+Tanh。
    # 21k pretrain+1k fine-tune 版会把 pre_logits 置为恒等 (kernel=I, bias=0)，
    # 最终 head 是直接 on encoder 特征训练的，不应再过 pre_logits+Tanh。
    has_pre_logits = "pre_logits/kernel" in d
    if has_pre_logits:
        pl_k = d["pre_logits/kernel"]
        pl_b = d["pre_logits/bias"]
        is_identity = (
            pl_k.shape == (hidden_dim, hidden_dim)
            and np.allclose(pl_k, np.eye(hidden_dim), atol=1e-4)
            and np.linalg.norm(pl_b) < 1e-4
        )
        if is_identity:
            has_pre_logits = False

    num_classes = len(class_indices)
    # -1 占位表示 synset 不在 21k 列表，对应行用零填充
    safe_idx = torch.tensor(
        [i if i >= 0 else 0 for i in class_indices], dtype=torch.long
    )
    head_w = head_w_full[safe_idx].clone()
    head_b = head_b_full[safe_idx].clone()
    for i, ci in enumerate(class_indices):
        if ci < 0:
            head_w[i].zero_()
            head_b[i].zero_()

    if has_pre_logits:
        pl_w = torch.tensor(d["pre_logits/kernel"].T, dtype=torch.float32)
        pl_b_t = torch.tensor(d["pre_logits/bias"], dtype=torch.float32)
        pre_logits_layer = nn.Linear(hidden_dim, hidden_dim)
        pre_logits_layer.weight.data.copy_(pl_w)
        pre_logits_layer.bias.data.copy_(pl_b_t)
        cls_layer = nn.Linear(hidden_dim, num_classes)
        cls_layer.weight.data.copy_(head_w)
        cls_layer.bias.data.copy_(head_b)
        model.head = nn.Sequential(pre_logits_layer, nn.Tanh(), cls_layer)
        print(
            f"  Head: Linear({hidden_dim}→{hidden_dim}) → Tanh → "
            f"Linear({hidden_dim}→{num_classes})"
        )
    else:
        cls_layer = nn.Linear(hidden_dim, num_classes)
        cls_layer.weight.data.copy_(head_w)
        cls_layer.bias.data.copy_(head_b)
        model.head = cls_layer
        print(f"  Head: Linear({hidden_dim}→{num_classes})")

    head_norm = sum(p.data.norm().item() for p in model.head.parameters())
    print(f"  Head weight norm: {head_norm:.4f}")

    prompt_count = 0
    for name, param in model.named_parameters():
        nl = name.lower()
        if "mask" in nl:
            continue
        do_init = (
            ("prompt" in nl and ("embeddings" in nl or "embedding" in nl))
            or ("fourier" in nl and "embedding" in nl)
            or ("qkv" in nl and "embedding" in nl)
            or ("random_vectors" in nl)
        )
        if do_init:
            nn.init.normal_(param, mean=0.0, std=prompt_init_std)
            prompt_count += param.numel()

    print(
        f"  Prompt token init: N(0, {prompt_init_std}) ({prompt_count} params)"
    )

    for param in model.head.parameters():
        param.requires_grad = False
    print("  Head frozen. Only prompt parameters trainable.")


class MultiViewTransform:
    """Return [center_crop] + N augmented views per image."""

    def __init__(self, cropsize, n_views, norm_params=None):
        if norm_params is None:
            norm_params = _VIT_NORM
        normalize = transforms.Normalize(**norm_params)
        self.center = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(cropsize),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.augment = transforms.Compose(
            [
                transforms.RandomResizedCrop(cropsize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.n_views = n_views

    def __call__(self, img):
        views = [self.center(img)]
        for _ in range(self.n_views):
            views.append(self.augment(img))
        return views


def build_dataloader(args: Any):
    """Build ImageFolder loader; expects data_dir, dataset, batch_size, cropsize, workers."""
    norm_params = get_vit_norm_params(getattr(args, "backbone", ""))
    normalize = transforms.Normalize(**norm_params)

    if args.dataset == "imagenet-c":
        data_path = os.path.join(
            args.data_dir, args.corruption, str(args.severity)
        )
    else:
        data_path = args.data_dir

    num_views = getattr(args, "num_views", 1)

    if num_views > 1:
        transform = MultiViewTransform(
            args.cropsize, n_views=num_views - 1, norm_params=norm_params
        )
        print(
            f"Loading dataset from: {data_path}  [multi-view: {num_views} views/image]"
        )
        dataset = datasets.ImageFolder(data_path, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.workers > 0,
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(args.cropsize),
                transforms.ToTensor(),
                normalize,
            ]
        )
        print(f"Loading dataset from: {data_path}")
        dataset = datasets.ImageFolder(data_path, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.workers > 0,
        )
    return loader, dataset
