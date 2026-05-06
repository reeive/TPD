#!/usr/bin/env python3
"""
run_tta_vpt.py - VPT test-time adaptation runner (zero-shot, single backbone weight).

Initializes the classification head directly from pre-trained backbone weights
(.npz), enabling zero-shot TTA without source-domain fine-tuning.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.configs.config import get_cfg
from src.models.build_model import build_model as _build_vpt_model
from src.tta import TPDTTAEngine, TTAEngine
from src.tta._common import (
    BACKBONE_REGISTRY,
    build_dataloader,
    build_head_indices,
    init_head_from_npz,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="VPT Test-Time Prompt Tuning")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_root", type=str, default="./checkpoints",
                        help="Root directory for backbone weights (.npz / timm cache)")
    parser.add_argument("--backbone", type=str, default="sup_vitb16_224")
    parser.add_argument("--num_tokens", type=int, default=5)
    parser.add_argument("--deep_prompt", action="store_true", default=True)
    parser.add_argument("--init_head", action="store_true", default=False)
    parser.add_argument("--prompt_init_std", type=float, default=0.02)

    parser.add_argument("--data-dir", type=str, default="./data/imagenet-r",
                        help="Dataset root (ImageFolder layout)")
    parser.add_argument("--dataset", type=str, default="imagenet-r",
                        choices=["imagenet-c", "imagenet-r", "imagenet-a",
                                 "imagenet", "folder", "CUB_200_2011"])
    parser.add_argument("--corruption", type=str, default="gaussian_noise")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--cropsize", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--protocol", type=str, default="online",
                        choices=["online", "episodic", "strict-episodic",
                                 "memory-episodic"])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tta_steps", type=int, default=1)
    parser.add_argument("--selection_p", type=float, default=0.1)
    parser.add_argument("--state_dim", type=int, default=16)
    parser.add_argument("--subspace_mode", type=str, default="svd",
                        choices=["svd", "random"],
                        help="Subspace basis: svd (paper) or random (fixed orthogonal)")
    parser.add_argument("--basis_window", type=int, default=None,
                        help="SVD basis history length (B). Default=W. Set B>=r to uncap SVD rank.")
    parser.add_argument("--adaptive_r", action="store_true", default=False,
                        help="Let SVD energy determine r (derivation's approach)")
    parser.add_argument("--max_state_dim", type=int, default=512,
                        help="Upper bound on r when adaptive_r is enabled")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--projection", type=str, default="pca",
                        choices=["pca", "random"])
    parser.add_argument("--c1_mode", type=str, default="full",
                        choices=["full", "diagonal", "auto"],
                        help="C1 Koopman: full, diagonal, or auto (full rho when well-conditioned)")
    parser.add_argument("--c1_auto_cond_threshold", type=float, default=1e8,
                        help="auto mode: max cond(Z_0^T Z_0) to use full EDMD for rho")
    parser.add_argument("--c1_threshold", type=float, default=1.0)
    parser.add_argument("--c1_ridge_lambda", type=float, default=0.01,
                        help="Ridge regularization for EDMD (lower → more sensitive)")
    parser.add_argument("--c1_rollback_patience", type=int, default=3)
    parser.add_argument("--c1_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--c2_energy", type=float, default=0.95)
    parser.add_argument("--snr_gate", action="store_true", default=False,
                        help="Enable SNR/consensus gating on learning rate")
    parser.add_argument("--snr_gate_tau", type=float, default=0.15,
                        help="Cosine threshold for SNR gate")
    parser.add_argument("--sb_gamma", type=float, default=1.0,
                        help="Spectral bridge C2 gamma exponent")
    parser.add_argument("--sb_beta", type=float, default=0.85,
                        help="Spectral bridge EMA decay")
    parser.add_argument("--sb_lambda_rK", type=float, default=0.5,
                        help="Spectral bridge Koopman residual weight")
    parser.add_argument("--cs_alpha", type=float, default=1.0,
                        help="Conflict-spectral C2: conflict penalty weight")
    parser.add_argument("--cs_beta_n", type=float, default=0.3,
                        help="Conflict-spectral C2: noise penalty weight")
    parser.add_argument("--cs_flip_L", type=int, default=5,
                        help="Conflict-spectral C2: FlipRate window length")
    parser.add_argument("--cr_buffer_decay", type=float, default=0.9,
                        help="Conflict-router C2: buffer decay rate")
    parser.add_argument("--cr_buffer_weight", type=float, default=0.3,
                        help="Conflict-router C2: buffer blending weight")
    parser.add_argument("--kh_gate", action="store_true", default=False,
                        help="Enable Koopman-Hankel risk gating")
    parser.add_argument("--kh_delay", type=int, default=2,
                        help="Hankel delay embedding length")
    parser.add_argument("--kh_gate_threshold", type=float, default=0.3,
                        help="Risk threshold for KH gate sigmoid")
    parser.add_argument("--c2_mode", type=str, default="qp_decomp",
                        choices=["fast", "full", "off", "koopman_bal", "spectral_bridge",
                                 "force_adaptive", "hankel_fa", "hbuo_peft",
                                 "a_soft", "diag_shrink",
                                 "conflict_spectral", "conflict_router", "qp_decomp"])
    parser.add_argument("--qp_kappa", type=float, default=2.0,
                        help="q/p C2: tanh sharpness on diagonal Koopman a_i")
    parser.add_argument("--qp_eps", type=float, default=1e-6,
                        help="q/p C2: numerical stability for q_i denominator")
    parser.add_argument("--hur_routing", type=str, default="fixed",
                        choices=["fixed", "adaptive_bn", "per_mode_bc"],
                        help="HUR routing mode")
    parser.add_argument("--hur_bn_max", type=float, default=0.5,
                        help="adaptive_bn: β_n when q=0 (max pass-through)")
    parser.add_argument("--hur_bn_min", type=float, default=0.05,
                        help="adaptive_bn: β_n when q=1 (min pass-through)")
    parser.add_argument("--hur_bc_tau", type=float, default=5.0,
                        help="per_mode_bc: sigmoid steepness for |a_i|-based β_c")
    parser.add_argument("--c2_variant", type=str, default="uncentered",
                        choices=["centered", "uncentered", "gated", "shrink"])
    parser.add_argument("--c2_gate_threshold", type=float, default=0.85)
    parser.add_argument("--c2_gate_blend", type=float, default=0.5)
    parser.add_argument("--c2_shrink_ratio", type=float, default=0.5)
    parser.add_argument("--gramian_interval", type=int, default=10)
    parser.add_argument("--kbal_warmup", type=int, default=3,
                        help="Koopman-Balanced C2 warmup steps (PCA fallback before balanced truncation)")
    parser.add_argument("--kbal_mu_beta", type=float, default=0.9)
    parser.add_argument("--kbal_update_freq", type=int, default=5)
    parser.add_argument("--kbal_gram_horizon", type=int, default=15)
    parser.add_argument("--kbal_shrink_min", type=float, default=0.05)
    parser.add_argument("--kbal_shrink_max", type=float, default=0.5)
    parser.add_argument("--kbal_fa_tau", type=float, default=5.0)

    # force_adaptive C2 sub-parameters
    parser.add_argument("--fa_mu_beta", type=float, default=0.9)
    parser.add_argument("--fa_shrink_min", type=float, default=0.05)
    parser.add_argument("--fa_shrink_max", type=float, default=0.8)
    parser.add_argument("--fa_energy_min", type=float, default=0.7)
    parser.add_argument("--fa_energy_max", type=float, default=0.99)
    parser.add_argument("--fa_tau", type=float, default=5.0)
    parser.add_argument("--fa_signal_proj_threshold", type=float, default=20.0)
    parser.add_argument("--fa_warmup", type=int, default=5)

    # hankel_fa C2 sub-parameters
    parser.add_argument("--hfa_mu_beta", type=float, default=0.9)
    parser.add_argument("--hfa_signal_proj_threshold", type=float, default=10.0)
    parser.add_argument("--hfa_warmup", type=int, default=5)
    parser.add_argument("--hfa_hankel_len", type=int, default=15)
    parser.add_argument("--hfa_hankel_energy", type=float, default=0.90)
    parser.add_argument("--hfa_update_freq", type=int, default=5)
    parser.add_argument("--hfa_shrink_noise", type=float, default=0.05)

    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--ensemble_alpha", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=0)

    parser.add_argument("--kdmd", action="store_true", default=False,
                        help="Enable KDMD dynamics-derived semantic anchors")
    parser.add_argument("--kdmd_dim", type=int, default=128,
                        help="RFF lifted dimension D for KDMD")
    parser.add_argument("--kdmd_lambda", type=float, default=1.0,
                        help="KDMD logit adjustment strength")
    parser.add_argument("--kdmd_gamma", type=float, default=1.0,
                        help="RBF kernel bandwidth for KDMD")
    parser.add_argument("--kdmd_temperature", type=float, default=1.0,
                        help="KDMD alignment score temperature")

    # KTMV: Koopman Trajectory Multi-View
    parser.add_argument("--ktmv", action="store_true", default=False,
                        help="Enable KTMV dynamics-based multi-view")
    parser.add_argument("--ktmv_views", type=int, default=4,
                        help="Number of virtual KTMV views")
    parser.add_argument("--ktmv_scale", type=float, default=0.1,
                        help="KTMV perturbation scale relative to ||z_t||")

    parser.add_argument("--num_views", type=int, default=1,
                        help="Multi-view TTA: number of augmented views per image (1=off)")
    parser.add_argument("--tune_head", action="store_true", default=False,
                        help="Also tune classification head during TTA")
    parser.add_argument("--tune_ln", action="store_true", default=False,
                        help="Also tune LayerNorm parameters during TTA")
    parser.add_argument("--head_lr_mult", type=float, default=0.01,
                        help="Head LR multiplier relative to prompt LR")
    parser.add_argument("--head_grad_clip", type=float, default=1.0,
                        help="Per-element gradient clipping for head params")

    # T2: Online Feature Prototypes
    parser.add_argument("--proto", action="store_true", default=False,
                        help="Enable online feature prototype rectification")
    parser.add_argument("--proto_gamma", type=float, default=1.0,
                        help="Prototype logit weight")
    parser.add_argument("--proto_tau", type=float, default=0.1,
                        help="Prototype cosine similarity temperature")
    parser.add_argument("--proto_momentum", type=float, default=0.9,
                        help="Prototype EMA momentum")
    parser.add_argument("--proto_conf", type=float, default=0.5,
                        help="Min confidence to update prototypes")

    # T3: Multi-objective loss
    parser.add_argument("--lambda_attn", type=float, default=0.0,
                        help="Attention entropy loss weight (0=off)")
    parser.add_argument("--lambda_pl", type=float, default=0.0,
                        help="Pseudo-label KL loss weight (0=off)")
    parser.add_argument("--pl_threshold", type=float, default=0.7,
                        help="Teacher confidence threshold for pseudo-labels")

    # T4: Koopman-SAM
    parser.add_argument("--sam", action="store_true", default=False,
                        help="Enable Koopman-SAM sharpness-aware adaptation")
    parser.add_argument("--sam_rho", type=float, default=0.05,
                        help="SAM perturbation radius")

    # T5: EMA Teacher-Student
    parser.add_argument("--ema", action="store_true", default=False,
                        help="Enable EMA teacher-student")
    parser.add_argument("--ema_alpha", type=float, default=0.999,
                        help="EMA decay rate")
    parser.add_argument("--ema_temperature", type=float, default=1.0,
                        help="EMA teacher temperature for KL divergence")

    # Anchor regularization
    parser.add_argument("--anchor_lambda", type=float, default=0.0,
                        help="Anchor regularization λ for ||P-P₀||² (0=off)")

    parser.add_argument("--output-dir", type=str, default="./tta_vpt_results")
    parser.add_argument("--engine", type=str, default="tpd",
                        choices=["tpd"],
                        help="Engine: 'tpd' (TPDTTAEngine)")
    parser.add_argument("--tpd-config", type=str, default="",
                        help="Optional YAML path for paper-aligned TPD settings")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def build_vpt_model(args, device):
    cfg = get_cfg()
    cfg.MODEL.TYPE = "vit"
    cfg.MODEL.TRANSFER_TYPE = "prompt"
    cfg.MODEL.MODEL_ROOT = args.model_root
    cfg.MODEL.PROMPT.NUM_TOKENS = args.num_tokens
    cfg.MODEL.PROMPT.DEEP = args.deep_prompt
    cfg.MODEL.PROMPT.DROPOUT = 0.0
    cfg.DATA.FEATURE = args.backbone
    cfg.DATA.NUMBER_CLASSES = args.num_classes
    cfg.DATA.CROPSIZE = args.cropsize
    cfg.NUM_GPUS = 1
    cfg.DBG = False

    model, _ = _build_vpt_model(cfg)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded VPT checkpoint: {args.checkpoint}")
    elif args.checkpoint:
        print(f"WARNING: checkpoint not found: {args.checkpoint}")

    if getattr(args, "init_head", False):
        data_path = args.data_dir
        if args.dataset == "imagenet-c":
            data_path = os.path.join(args.data_dir, args.corruption, str(args.severity))
        print("Building head index mapping...")
        class_indices, n_cls = build_head_indices(data_path, backbone=args.backbone)
        npz_filename = BACKBONE_REGISTRY.get(
            args.backbone, (f"{args.backbone}.npz", "1k"))[0]
        npz_path = os.path.join(args.model_root, npz_filename)
        init_head_from_npz(model, npz_path, class_indices, device,
                           prompt_init_std=args.prompt_init_std)

    model = model.to(device)
    model.eval()
    return model


def build_engine_config(args):
    return {
        "protocol": args.protocol,
        "num_classes": args.num_classes,
        "tta": {
            "state": {
                "dim": args.state_dim,
                "projection": args.projection,
                "adaptive_r": getattr(args, "adaptive_r", False),
                "max_state_dim": getattr(args, "max_state_dim", 512),
            },
            "koopman": {
                "window": args.window_size,
                "ridge_lambda": args.c1_ridge_lambda,
                "rollback_patience": args.c1_rollback_patience,
                "rho_threshold": args.c1_threshold,
                "min_lr_ratio": args.c1_min_lr_ratio,
                "c1_mode": args.c1_mode,
                "auto_cond_threshold": getattr(
                    args, "c1_auto_cond_threshold", 1e8),
            },
            "hbuo": {
                "mode": args.c2_mode,
                "variant": args.c2_variant,
                "gate_threshold": args.c2_gate_threshold,
                "gate_blend": args.c2_gate_blend,
                "shrink_ratio": args.c2_shrink_ratio,
                "num_perturbations": 5,
                "perturbation_scale": 0.01,
                "energy_threshold": args.c2_energy,
                "update_freq": args.gramian_interval,
                "snr_gate": {
                    "enabled": getattr(args, "snr_gate", False),
                    "tau_cos": getattr(args, "snr_gate_tau", 0.15),
                    "steepness": 5.0,
                },
                "spectral_bridge": {
                    "beta": getattr(args, "sb_beta", 0.85),
                    "lambda_rK": getattr(args, "sb_lambda_rK", 0.5),
                    "gamma": getattr(args, "sb_gamma", 1.0),
                    "w_min": 0.05,
                },
                "kh_gate": {
                    "enabled": getattr(args, "kh_gate", False),
                    "delay_L": getattr(args, "kh_delay", 2),
                    "ridge": 1e-3,
                    "window": 15,
                    "gate_scale": 5.0,
                    "gate_threshold": getattr(args, "kh_gate_threshold", 0.3),
                },
                "a_soft": {
                    "beta": 0.9,
                    "alpha_base": 0.0,
                    "alpha_slope": 0.3,
                    "snr_ref": 1.0,
                    "alpha_max": 0.6,
                },
                "conflict_spectral": {
                    "beta_ema": 0.85,
                    "flip_L": getattr(args, "cs_flip_L", 5),
                    "alpha": getattr(args, "cs_alpha", 1.0),
                    "beta_n": getattr(args, "cs_beta_n", 0.3),
                },
                "conflict_router": {
                    "tau_s": 3.0,
                    "tau_c": 3.0,
                    "tau_n": 3.0,
                    "buffer_decay": getattr(args, "cr_buffer_decay", 0.9),
                    "buffer_weight": getattr(args, "cr_buffer_weight", 0.3),
                },
                "qp_decomp": {
                    "kappa": getattr(args, "qp_kappa", 2.0),
                    "eps": getattr(args, "qp_eps", 1e-6),
                    "buffer_decay": getattr(args, "cr_buffer_decay", 0.9),
                    "buffer_weight": getattr(args, "cr_buffer_weight", 0.3),
                },
                "koopman_bal": {
                    "warmup": getattr(args, "kbal_warmup", 3),
                    "mu_beta": getattr(args, "kbal_mu_beta", 0.9),
                    "update_freq": getattr(args, "kbal_update_freq", 5),
                    "gram_horizon": getattr(args, "kbal_gram_horizon", 15),
                    "shrink_min": getattr(args, "kbal_shrink_min", 0.05),
                    "shrink_max": getattr(args, "kbal_shrink_max", 0.5),
                    "fa_tau": getattr(args, "kbal_fa_tau", 5.0),
                },
                "force_adaptive": {
                    "mu_beta": getattr(args, "fa_mu_beta", 0.9),
                    "shrink_min": getattr(args, "fa_shrink_min", 0.05),
                    "shrink_max": getattr(args, "fa_shrink_max", 0.8),
                    "energy_min": getattr(args, "fa_energy_min", 0.7),
                    "energy_max": getattr(args, "fa_energy_max", 0.99),
                    "tau": getattr(args, "fa_tau", 5.0),
                    "signal_proj_threshold": getattr(args, "fa_signal_proj_threshold", 20.0),
                    "warmup": getattr(args, "fa_warmup", 5),
                },
                "hankel_fa": {
                    "mu_beta": getattr(args, "hfa_mu_beta", 0.9),
                    "signal_proj_threshold": getattr(args, "hfa_signal_proj_threshold", 10.0),
                    "warmup": getattr(args, "hfa_warmup", 5),
                    "hankel_len": getattr(args, "hfa_hankel_len", 15),
                    "hankel_energy": getattr(args, "hfa_hankel_energy", 0.90),
                    "update_freq": getattr(args, "hfa_update_freq", 5),
                    "shrink_noise": getattr(args, "hfa_shrink_noise", 0.05),
                },
            },
            "kdmd": {
                "enabled": getattr(args, "kdmd", False),
                "dim": getattr(args, "kdmd_dim", 128),
                "gamma": getattr(args, "kdmd_gamma", 1.0),
                "lam": getattr(args, "kdmd_lambda", 1.0),
                "temperature": getattr(args, "kdmd_temperature", 1.0),
            },
            "ktmv": {
                "enabled": getattr(args, "ktmv", False),
                "n_views": getattr(args, "ktmv_views", 4),
                "scale": getattr(args, "ktmv_scale", 0.1),
            },
            "update": {
                "lr": args.lr,
                "steps_per_sample": args.tta_steps,
                "selection_p": args.selection_p,
                "confidence_threshold": args.confidence_threshold,
                "ensemble_alpha": args.ensemble_alpha,
            },
            "scope": {
                "tune_head": getattr(args, "tune_head", False),
                "tune_ln": getattr(args, "tune_ln", False),
                "head_lr_mult": getattr(args, "head_lr_mult", 0.01),
                "head_grad_clip": getattr(args, "head_grad_clip", 1.0),
            },
            "prototypes": {
                "enabled": getattr(args, "proto", False),
                "gamma": getattr(args, "proto_gamma", 1.0),
                "tau": getattr(args, "proto_tau", 0.1),
                "momentum": getattr(args, "proto_momentum", 0.9),
                "conf_threshold": getattr(args, "proto_conf", 0.5),
            },
            "loss": {
                "lambda_attn": getattr(args, "lambda_attn", 0.0),
                "lambda_pl": getattr(args, "lambda_pl", 0.0),
                "pl_threshold": getattr(args, "pl_threshold", 0.7),
            },
            "sam": {
                "enabled": getattr(args, "sam", False),
                "rho": getattr(args, "sam_rho", 0.05),
            },
            "ema": {
                "enabled": getattr(args, "ema", False),
                "alpha": getattr(args, "ema_alpha", 0.999),
                "temperature": getattr(args, "ema_temperature", 1.0),
            },
            "anchor": {
                "lambda": getattr(args, "anchor_lambda", 0.0),
            },
        },
    }


def maybe_attach_tpd_config(cfg, args):
    yaml_path = args.tpd_config
    if not yaml_path:
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "vit_tpd.yaml")
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            yaml_path = candidate

    if not yaml_path or not os.path.isfile(yaml_path):
        return cfg

    with open(yaml_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if "tpd" in loaded:
        cfg["tpd"] = loaded["tpd"]
    return cfg


def run_tta(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Building VPT model...")
    model = build_vpt_model(args, device)

    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable before engine setup: {trainable_before:,} / {total_params:,}")

    cfg = build_engine_config(args)
    use_tpd = getattr(args, "engine", "tpd") == "tpd"

    if use_tpd:
        cfg = maybe_attach_tpd_config(cfg, args)
        tpd_sub = cfg.setdefault("tpd", {}).setdefault("subspace", {})
        tpd_sub["projection_mode"] = getattr(args, "subspace_mode", "svd")
        tpd_sub["rank"] = args.state_dim
        bw = getattr(args, "basis_window", None)
        if bw is not None:
            tpd_sub["basis_window"] = bw
        tpd_hur = cfg["tpd"].setdefault("hur", {})
        tpd_hur["routing_mode"] = getattr(args, "hur_routing", "fixed")
        tpd_hur["bn_max"] = getattr(args, "hur_bn_max", 0.5)
        tpd_hur["bn_min"] = getattr(args, "hur_bn_min", 0.05)
        tpd_hur["bc_tau"] = getattr(args, "hur_bc_tau", 5.0)

    # Auto-detect feature dimension from model for prototypes
    feat_dim = None
    if hasattr(model, 'head'):
        head = model.head
        if isinstance(head, nn.Sequential):
            for m in reversed(list(head.children())):
                if isinstance(m, nn.Linear):
                    feat_dim = m.in_features
                    break
        elif isinstance(head, nn.Linear):
            feat_dim = head.in_features
    if feat_dim and "tta" in cfg and "prototypes" in cfg["tta"]:
        cfg["tta"]["prototypes"]["feat_dim"] = feat_dim
    elif feat_dim:
        cfg.setdefault("feat_dim", feat_dim)

    if use_tpd:
        engine = TPDTTAEngine(model=model, cfg=cfg, device=device)
    else:
        engine = TTAEngine(model=model, cfg=cfg, device=device)

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after engine setup: {trainable_after:,} / {total_params:,}")

    print("Building dataset...")
    loader, dataset = build_dataloader(args)
    num_samples = args.num_samples if args.num_samples > 0 else len(dataset)

    c2_desc = f"{args.c2_mode}"
    if args.c2_mode == "fast":
        c2_desc += f"/{args.c2_variant}"

    engine_label = "TPDTTAEngine" if use_tpd else "TTAEngine (baseline)"
    print(f"\n{'='*64}")
    print(f"VPT Test-Time Prompt Tuning  [{engine_label}]")
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Protocol: {args.protocol}")
    print(f"C2 mode: {c2_desc}")
    kdmd_desc = (f"ON (D={args.kdmd_dim}, λ={args.kdmd_lambda}, "
                 f"γ={args.kdmd_gamma}, T={args.kdmd_temperature})"
                 if args.kdmd else "OFF")
    print(f"KDMD: {kdmd_desc}")
    ktmv_desc = (f"ON (views={args.ktmv_views}, scale={args.ktmv_scale})"
                 if args.ktmv else "OFF")
    print(f"KTMV: {ktmv_desc}")
    scope_parts = ["prompt"]
    if getattr(args, "tune_head", False):
        scope_parts.append(f"head(lr_m={args.head_lr_mult})")
    if getattr(args, "tune_ln", False):
        scope_parts.append("LN")
    nv = getattr(args, "num_views", 1)
    print(f"Scope: {'+'.join(scope_parts)} | Views: {nv} | Tokens: {args.num_tokens}")
    r_desc = f"adaptive(max={args.max_state_dim})" if getattr(args, "adaptive_r", False) else str(args.state_dim)
    sub_mode = getattr(args, "subspace_mode", "svd")
    bw = getattr(args, "basis_window", None)
    bw_desc = f" | B: {bw}" if bw is not None else ""
    anchor_desc = f" | anchor: {args.anchor_lambda}" if args.anchor_lambda > 0 else ""
    print(f"LR: {args.lr} | K: {args.tta_steps} | W: {args.window_size} | C1: {args.c1_mode} | r: {r_desc} | proj: {sub_mode}{bw_desc}{anchor_desc}")
    # Optional TTA features
    v4_parts = []
    if getattr(args, "proto", False):
        v4_parts.append(f"Proto(γ={args.proto_gamma},τ={args.proto_tau})")
    if getattr(args, "lambda_attn", 0) > 0:
        v4_parts.append(f"AttnEnt(λ={args.lambda_attn})")
    if getattr(args, "lambda_pl", 0) > 0:
        v4_parts.append(f"PL(λ={args.lambda_pl},thr={args.pl_threshold})")
    if getattr(args, "sam", False):
        v4_parts.append(f"K-SAM(ρ={args.sam_rho})")
    if getattr(args, "ema", False):
        v4_parts.append(f"EMA(α={args.ema_alpha})")
    if v4_parts:
        print(f"V4: {' | '.join(v4_parts)}")
    print(f"{'='*64}\n")

    cudnn.benchmark = True
    t_start = time.time()
    total_correct = 0
    total_seen = 0
    all_records = []
    num_rollbacks = 0
    num_views = getattr(args, "num_views", 1)

    for batch_idx, (images, labels) in enumerate(loader):
        if total_seen >= num_samples:
            break

        if isinstance(images, list):
            # Multi-view: images = [center_crop, aug1, ..., augN], each (1, C, H, W)
            center_images = images[0].to(device)
            all_views = torch.cat(images, dim=0).to(device)
            labels = labels.to(device)
            preds, info = engine.adapt_and_predict(
                center_images, labels, x_adapt=all_views)
        else:
            images = images.to(device)
            labels = labels.to(device)
            preds, info = engine.adapt_and_predict(images, labels)

        batch_correct = (preds == labels).sum().item()
        total_correct += batch_correct
        total_seen += len(labels)

        if info.get("rollback", False):
            num_rollbacks += 1

        all_records.append({
            "batch": batch_idx,
            "accuracy": batch_correct / len(labels),
            "rho": info.get("rho", 0.0),
            "eta": info.get("eta", 0.0),
            "entropy": info.get("entropy", 0.0),
            "drift": info.get("drift", 0.0),
            "rollback": info.get("rollback", False),
            "skipped": info.get("skipped", False),
            "mode": info.get("mode", ""),
            "stable": info.get("stable", True),
            "q_mean": info.get("q_mean", 0.0),
            "p_mean": info.get("p_mean", 0.0),
            "u_raw_norm": info.get("u_raw_norm", 0.0),
            "u_applied_norm": info.get("u_applied_norm", 0.0),
        })

        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            acc_pct = total_correct / total_seen * 100
            print(f"[batch {batch_idx+1:4d}] acc={acc_pct:6.2f}% "
                  f"rho={info.get('rho', 0):.4f} "
                  f"eta={info.get('eta', 0):.6f} "
                  f"drift={info.get('drift', 0):.4f} "
                  f"u_raw={info.get('u_raw_norm', 0):.6f} "
                  f"u_app={info.get('u_applied_norm', 0):.6f} "
                  f"q={info.get('q_mean', 0):.4f} "
                  f"p={info.get('p_mean', 0):.4f} "
                  f"mode={info.get('mode', '?')} "
                  f"rb={num_rollbacks} "
                  f"skip={1 if info.get('skipped') else 0}")

    elapsed = time.time() - t_start
    final_acc = total_correct / max(total_seen, 1) * 100
    rho_vals = [r["rho"] for r in all_records]
    mean_rho = float(np.mean(rho_vals)) if rho_vals else 0
    max_rho = float(np.max(rho_vals)) if rho_vals else 0
    mean_drift = float(np.mean([r["drift"] for r in all_records])) if all_records else 0

    print(f"\n{'='*64}")
    print(f"VPT TTA run complete")
    print(f"Accuracy: {final_acc:.2f}% ({total_correct}/{total_seen})")
    print(f"Time: {elapsed:.1f}s")
    print(f"Mean rho: {mean_rho:.4f} | Max rho: {max_rho:.4f}")
    print(f"Mean drift: {mean_drift:.4f}")
    print(f"{'='*64}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "accuracy": final_acc,
        "total_samples": total_seen,
        "time_sec": elapsed,
        "mean_rho": mean_rho,
        "koopman_rho_max": max_rho,
        "mean_drift": mean_drift,
        "num_rollbacks": num_rollbacks,
        "rollback_count": num_rollbacks,
        "engine": "tpd",
        "c1_mode": args.c1_mode,
        "state_dim": args.state_dim,
        "anchor_lambda": args.anchor_lambda,
        "protocol": args.protocol,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "lr": args.lr,
        "tta_steps": args.tta_steps,
        "c2_mode": args.c2_mode,
        "c2_variant": args.c2_variant if args.c2_mode == "fast" else "N/A",
        "kdmd": args.kdmd,
        "kdmd_lambda": args.kdmd_lambda if args.kdmd else "N/A",
        "ktmv": getattr(args, "ktmv", False),
        "ktmv_views": getattr(args, "ktmv_views", 4) if getattr(args, "ktmv", False) else "N/A",
        "ktmv_scale": getattr(args, "ktmv_scale", 0.1) if getattr(args, "ktmv", False) else "N/A",
        "num_views": getattr(args, "num_views", 1),
        "num_tokens": args.num_tokens,
        "tune_head": getattr(args, "tune_head", False),
        "tune_ln": getattr(args, "tune_ln", False),
        "head_lr_mult": getattr(args, "head_lr_mult", 0.01),
        "proto": getattr(args, "proto", False),
        "proto_gamma": getattr(args, "proto_gamma", 1.0),
        "lambda_attn": getattr(args, "lambda_attn", 0.0),
        "lambda_pl": getattr(args, "lambda_pl", 0.0),
        "sam": getattr(args, "sam", False),
        "sam_rho": getattr(args, "sam_rho", 0.05),
        "ema": getattr(args, "ema", False),
        "ema_alpha": getattr(args, "ema_alpha", 0.999),
    }
    with open(os.path.join(args.output_dir, "tta_vpt_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, "tta_vpt_batch_records.json"), "w") as f:
        json.dump(all_records, f)
    print(f"Results saved to {args.output_dir}/tta_vpt_results.json")
    print(f"Batch records saved to {args.output_dir}/tta_vpt_batch_records.json")


if __name__ == "__main__":
    run_tta(parse_args())
