from __future__ import annotations

import copy
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import yaml

from .tpd_runtime import TPDStepResult, TPDRuntime


_DEFAULT_VIT_TPD_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "vit_tpd.yaml")
)


def _deep_merge_dict(base: Dict, override: Dict) -> Dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_config(config_path: Optional[str]) -> Dict:
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"TPD config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"TPD config must load to a dict: {config_path}")
    return loaded


def _extract_tpd_config(cfg: Optional[Dict]) -> Dict:
    cfg = cfg or {}
    if not isinstance(cfg, dict):
        return {}

    config_path = cfg.get("tpd_config_path") or cfg.get("tpd_config")
    base = _load_yaml_config(config_path or _DEFAULT_VIT_TPD_CONFIG)

    if "tpd" in base:
        base = base["tpd"]

    explicit_tpd = cfg.get("tpd", {})
    merged = _deep_merge_dict(base, explicit_tpd)

    if "tta" in cfg:
        update_cfg = cfg.get("tta", {}).get("update", {})
        protocol = cfg.get("protocol", merged.get("protocol", "online"))
        merged = _deep_merge_dict(
            merged,
            {
                "protocol": protocol,
                "base_lr": update_cfg.get("lr", merged.get("base_lr", 1e-3)),
                "update": {
                    "steps_per_sample": update_cfg.get(
                        "steps_per_sample",
                        merged.get("update", {}).get("steps_per_sample", 1),
                    ),
                    "selection_p": update_cfg.get(
                        "selection_p",
                        merged.get("update", {}).get("selection_p", 1.0),
                    ),
                },
                "subspace": {
                    "rank": cfg.get("tta", {}).get(
                        "state", {}
                    ).get("dim", merged.get("subspace", {}).get("rank", 16)),
                    "window_size": cfg.get("tta", {}).get(
                        "koopman", {}
                    ).get("window", merged.get("subspace", {}).get("window_size", 10)),
                    "min_history": merged.get("subspace", {}).get("min_history", 2),
                },
                "akc": {
                    "mode": cfg.get("tta", {}).get(
                        "koopman", {}
                    ).get("c1_mode", merged.get("akc", {}).get("mode", "auto")),
                    "rho_threshold": cfg.get("tta", {}).get(
                        "koopman", {}
                    ).get("rho_threshold", merged.get("akc", {}).get("rho_threshold", 1.0)),
                    "rollback_patience": cfg.get("tta", {}).get(
                        "koopman", {}
                    ).get(
                        "rollback_patience",
                        merged.get("akc", {}).get("rollback_patience", 3),
                    ),
                    "min_scale": cfg.get("tta", {}).get(
                        "koopman", {}
                    ).get("min_lr_ratio", merged.get("akc", {}).get("min_scale", 0.1)),
                    "full_condition_threshold": cfg.get("tta", {}).get(
                        "koopman", {}
                    ).get(
                        "auto_cond_threshold",
                        merged.get("akc", {}).get("full_condition_threshold", 1e5),
                    ),
                },
                "hur": {
                    "kappa": cfg.get("tta", {}).get(
                        "hbuo", {}
                    ).get("qp_decomp", {}).get(
                        "kappa", merged.get("hur", {}).get("kappa", 2.0)
                    ),
                    "eps": cfg.get("tta", {}).get(
                        "hbuo", {}
                    ).get("qp_decomp", {}).get(
                        "eps", merged.get("hur", {}).get("eps", 1e-6)
                    ),
                },
            },
        )

    return merged


def _select_confident_logits(logits: torch.Tensor, top_ratio: float) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] <= 1:
        return logits
    ratio = float(top_ratio)
    if ratio >= 1.0:
        return logits
    ratio = max(ratio, 1.0 / float(logits.shape[0]))
    entropy = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)
    keep = max(1, int(round(logits.shape[0] * ratio)))
    _, idx = entropy.topk(keep, largest=False)
    return logits[idx]


def _avg_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1).mean()


class TPDTTAEngine(TPDRuntime):
    """Runner-facing ViT engine over the paper-aligned TPD runtime."""

    def __init__(self, model: nn.Module, cfg: Optional[Dict], device: Optional[torch.device] = None):
        tpd_cfg = _extract_tpd_config(cfg)
        subspace_cfg = tpd_cfg.get("subspace", {})
        akc_cfg = tpd_cfg.get("akc", {})
        hur_cfg = tpd_cfg.get("hur", {})
        update_cfg = tpd_cfg.get("update", {})

        self.cfg = cfg or {}
        self.tpd_cfg = tpd_cfg
        self.protocol = tpd_cfg.get("protocol", self.cfg.get("protocol", "online"))
        self.steps_per_sample = int(update_cfg.get("steps_per_sample", 1))
        self.selection_p = float(update_cfg.get("selection_p", 1.0))
        self._metrics = []

        super().__init__(
            model=model,
            base_lr=float(tpd_cfg.get("base_lr", 1e-3)),
            subspace_rank=int(subspace_cfg.get("rank", 16)),
            window_size=int(subspace_cfg.get("window_size", 10)),
            min_history=int(subspace_cfg.get("min_history", 2)),
            akc_mode=str(akc_cfg.get("mode", "auto")),
            rho_threshold=float(akc_cfg.get("rho_threshold", 1.0)),
            rollback_patience=int(akc_cfg.get("rollback_patience", 3)),
            min_scale=float(akc_cfg.get("min_scale", 0.1)),
            full_condition_threshold=float(akc_cfg.get("full_condition_threshold", 1e5)),
            full_improvement_margin=float(akc_cfg.get("full_improvement_margin", 0.05)),
            min_mode_dwell=int(akc_cfg.get("min_mode_dwell", 2)),
            hur_beta_c=float(hur_cfg.get("beta_c", 0.25)),
            hur_beta_n=float(hur_cfg.get("beta_n", 0.1)),
            hur_kappa=float(hur_cfg.get("kappa", 2.0)),
            hur_eps=float(hur_cfg.get("eps", 1e-6)),
            hur_routing_mode=str(hur_cfg.get("routing_mode", "fixed")),
            hur_bn_max=float(hur_cfg.get("bn_max", 0.5)),
            hur_bn_min=float(hur_cfg.get("bn_min", 0.05)),
            hur_bc_tau=float(hur_cfg.get("bc_tau", 5.0)),
            projection_mode=str(subspace_cfg.get("projection_mode", "svd")),
            basis_window=subspace_cfg.get("basis_window", None),
            anchor_lambda=float(self.cfg.get("tta", {}).get("anchor", {}).get("lambda", 0.0)),
            device=device,
        )
        self._freeze_non_prompt_parameters()
        self.model.eval()

    def _freeze_non_prompt_parameters(self) -> None:
        prompt_ids = {id(param) for param in self.prompt_adapter.parameters()}
        for param in self.model.parameters():
            param.requires_grad_(id(param) in prompt_ids)

    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        confident_logits = _select_confident_logits(logits, self.selection_p)
        return _avg_entropy(confident_logits)

    def _build_info(self, logits: torch.Tensor, step_result: Optional[TPDStepResult], loss_value: float, skipped: bool) -> Dict:
        drift = float((self.prompt_vector() - self._initial_prompt.vector.to(self.device)).norm().item())
        info = {
            "entropy": float(loss_value),
            "rho": 0.0,
            "eta": self.base_lr,
            "drift": drift,
            "rollback": False,
            "skipped": skipped,
            "mode": "warmup",
            "stable": True,
            "q_mean": 0.0,
            "p_mean": 0.0,
            "u_raw_norm": 0.0,
            "u_applied_norm": 0.0,
        }
        if step_result is not None:
            info.update(
                {
                    "rho": float(step_result.rho),
                    "eta": float(self.base_lr * step_result.scale),
                    "rollback": bool(step_result.rollback),
                    "mode": step_result.mode,
                    "stable": bool(step_result.stable),
                    "q_mean": float(step_result.q_mean),
                    "p_mean": float(step_result.p_mean),
                    "u_raw_norm": float(step_result.u_raw_norm),
                    "u_applied_norm": float(step_result.u_applied_norm),
                }
            )
        return info

    def adapt_and_predict(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, x_adapt: Optional[torch.Tensor] = None):
        if "episodic" in str(self.protocol):
            self.reset_state()

        adapt_input = x_adapt if x_adapt is not None else x
        last_result: Optional[TPDStepResult] = None
        skipped = False
        loss_value = 0.0

        for _ in range(max(self.steps_per_sample, 1)):
            logits_adapt = self.model(adapt_input)
            if isinstance(logits_adapt, (tuple, list)):
                logits_adapt = logits_adapt[0]
            loss = self._entropy_loss(logits_adapt)
            if not torch.isfinite(loss):
                skipped = True
                break
            loss_value = float(loss.item())
            last_result = self.step_from_loss(loss, lr=self.base_lr)
            if last_result.rollback:
                break

        with torch.no_grad():
            logits = self.model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            preds = logits.argmax(dim=-1)

        info = self._build_info(logits, last_result, loss_value, skipped)
        if y is not None:
            info["accuracy"] = float((preds == y).float().mean().item())
        self._metrics.append(info.copy())
        return preds, info

    def get_summary(self) -> Dict:
        if not self._metrics:
            return {
                "total_samples": 0,
                "entropy_mean": 0.0,
                "rho_mean": 0.0,
                "drift_mean": 0.0,
                "accuracy_mean": 0.0,
            }

        def _mean(key: str) -> float:
            return float(sum(item.get(key, 0.0) for item in self._metrics) / len(self._metrics))

        return {
            "total_samples": len(self._metrics),
            "entropy_mean": _mean("entropy"),
            "rho_mean": _mean("rho"),
            "drift_mean": _mean("drift"),
            "accuracy_mean": _mean("accuracy"),
            "rollback_count": int(sum(1 for item in self._metrics if item.get("rollback", False))),
        }
