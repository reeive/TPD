import os
import sys

import torch
import torch.nn as nn
import yaml
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.tta.tpd_engine import TPDTTAEngine, _extract_tpd_config


class DummyViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.zeros(4))
        self.head = nn.Linear(3, 2)

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        prompt_bias = self.prompt_embeddings[:2].sum()
        return self.head(pooled) + prompt_bias


def _build_cfg():
    return {
        "protocol": "online",
        "tta": {
            "state": {"dim": 2},
            "koopman": {
                "window": 4,
                "rho_threshold": 1.0,
                "rollback_patience": 2,
                "min_lr_ratio": 0.1,
                "c1_mode": "auto",
                "auto_cond_threshold": 1e6,
            },
            "update": {"lr": 0.1, "steps_per_sample": 1, "selection_p": 1.0},
            "hbuo": {"qp_decomp": {"buffer_weight": 0.25, "kappa": 2.0, "eps": 1e-6}},
        },
    }


def test_tpd_engine_adapt_and_predict_matches_runner_contract():
    model = DummyViTModel()
    engine = TPDTTAEngine(model=model, cfg=_build_cfg(), device=torch.device("cpu"))
    x = torch.randn(2, 3, 4, 4)
    y = torch.tensor([0, 1])

    preds, info = engine.adapt_and_predict(x, y)

    assert preds.shape == y.shape
    for key in ["rho", "eta", "entropy", "drift", "rollback", "skipped"]:
        assert key in info


def test_tpd_engine_merges_yaml_with_runner_overrides():
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "vit_tpd.yaml")
        with open(yaml_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "tpd": {
                        "protocol": "episodic",
                        "base_lr": 0.05,
                        "subspace": {"rank": 4, "window_size": 6, "min_history": 2},
                        "akc": {"mode": "diagonal", "rho_threshold": 0.9},
                        "update": {"steps_per_sample": 3, "selection_p": 0.75},
                    }
                },
                handle,
            )

        cfg = {
            "protocol": "online",
            "tpd_config_path": yaml_path,
            "tta": {
                "state": {"dim": 2},
                "koopman": {
                    "window": 4,
                    "rho_threshold": 1.25,
                    "rollback_patience": 5,
                    "min_lr_ratio": 0.2,
                    "c1_mode": "auto",
                    "auto_cond_threshold": 321.0,
                },
                "update": {"lr": 0.1, "steps_per_sample": 1, "selection_p": 1.0},
                "hbuo": {"qp_decomp": {"kappa": 3.0, "eps": 1e-4}},
            },
        }

        tpd_cfg = _extract_tpd_config(cfg)
        assert tpd_cfg["protocol"] == "online"
        assert tpd_cfg["base_lr"] == 0.1
        assert tpd_cfg["subspace"]["rank"] == 2
        assert tpd_cfg["subspace"]["window_size"] == 4
        assert tpd_cfg["akc"]["mode"] == "auto"
        assert tpd_cfg["akc"]["rho_threshold"] == 1.25
        assert tpd_cfg["akc"]["rollback_patience"] == 5
        assert tpd_cfg["akc"]["min_scale"] == 0.2
        assert tpd_cfg["akc"]["full_condition_threshold"] == 321.0
        assert tpd_cfg["update"]["steps_per_sample"] == 1
        assert tpd_cfg["update"]["selection_p"] == 1.0
        assert tpd_cfg["hur"]["kappa"] == 3.0
        assert tpd_cfg["hur"]["eps"] == 1e-4
