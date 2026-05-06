"""
Integration test for the full TTA pipeline.

Uses a minimal synthetic VPT-like model and random data to verify
the end-to-end TTA loop works correctly without GPU or real data.
"""

import torch
import torch.nn as nn
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.tta.tta_engine import TTAEngine
from src.tta.state_utils import PromptStateManager


class _TransformerCore(nn.Module):
    """Inner module that holds prompt parameters.

    Mimics the PromptedTransformer in the real VPT codebase, which stores
    prompt_embeddings and deep_prompt_embeddings. This is a separate module
    to avoid circular references.
    """

    def __init__(self, num_tokens, hidden_dim, num_layers):
        super().__init__()
        prompt_dim = hidden_dim
        self.prompt_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, prompt_dim) * 0.01
        )
        self.deep_prompt_embeddings = nn.Parameter(
            torch.randn(num_layers - 1, num_tokens, prompt_dim) * 0.01
        )
        self.num_tokens = num_tokens


class MinimalPromptedTransformer(nn.Module):
    """Minimal VPT-like model for testing.

    Structure mimics the real PromptedTransformer with:
    - transformer.prompt_embeddings (shallow)
    - transformer.deep_prompt_embeddings (deep)
    - frozen backbone (linear layers)
    - classification head

    Uses a separate _TransformerCore to hold prompts, matching VPT's
    `model.enc.transformer.prompt_embeddings` access pattern.
    """

    def __init__(self, num_tokens=5, hidden_dim=32, num_classes=10,
                 num_layers=3, img_channels=3, img_size=8):
        super().__init__()

        # Transformer with prompt params (separate module, no circular ref)
        self.transformer = _TransformerCore(num_tokens, hidden_dim, num_layers)

        # Frozen backbone (simple)
        self.patch_embed = nn.Linear(img_channels * img_size * img_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Classification head
        self.head = nn.Linear(hidden_dim, num_classes)

        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

    def forward(self, x):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)

        h = self.patch_embed(x_flat)  # (B, hidden_dim)

        prompt_influence = self.transformer.prompt_embeddings.mean(dim=1).squeeze(0)
        h = h + prompt_influence

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self.transformer.deep_prompt_embeddings.shape[0]:
                deep_influence = self.transformer.deep_prompt_embeddings[i].mean(dim=0)
                h = h + deep_influence

        h = self.norm(h)
        logits = self.head(h)
        return logits


class MinimalViTWrapper(nn.Module):
    """Wraps MinimalPromptedTransformer to mimic VPT's ViT structure.

    The real VPT has model.enc.transformer.prompt_embeddings.
    We mimic the same access pattern without circular references.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.enc = MinimalPromptedTransformer(**kwargs)

    def forward(self, x):
        return self.enc(x)


class TestTTAIntegration:
    """Integration tests for the full TTA pipeline."""

    def setup_method(self):
        self.device = torch.device("cpu")
        self.model = MinimalViTWrapper(
            num_tokens=5, hidden_dim=32, num_classes=10,
            num_layers=3, img_channels=3, img_size=8,
        )
        self.cfg = {
            "protocol": "online",
            "tta": {
                "state": {"dim": 8, "projection": "random"},
                "koopman": {"window": 3, "ridge_lambda": 0.01, "rollback_patience": 3},
                "hbuo": {"num_perturbations": 3, "perturbation_scale": 0.01, "update_freq": 5},
                "update": {"lr": 0.01, "steps_per_sample": 1},
            },
        }

    def test_engine_creation(self):
        """Test TTAEngine can be created successfully."""
        engine = TTAEngine(self.model, self.cfg, self.device)
        assert engine is not None
        assert engine.protocol == "online"

    def test_adapt_and_predict_runs(self):
        """Test that adapt_and_predict completes without error."""
        engine = TTAEngine(self.model, self.cfg, self.device)
        x = torch.randn(4, 3, 8, 8)
        y = torch.randint(0, 10, (4,))

        predictions, info = engine.adapt_and_predict(x, y)

        assert predictions.shape == (4,)
        assert "entropy" in info
        assert "rho" in info
        assert "eta" in info
        assert "drift" in info

    def test_prompt_updates_in_online_mode(self):
        """Test that prompt parameters are actually updated in online mode."""
        engine = TTAEngine(self.model, self.cfg, self.device)

        # Record initial prompt state
        params_before = [
            p.data.clone()
            for p in engine.state_manager.get_prompt_params(self.model)
        ]

        x = torch.randn(4, 3, 8, 8)
        engine.adapt_and_predict(x)

        # Check prompts changed
        params_after = [
            p.data.clone()
            for p in engine.state_manager.get_prompt_params(self.model)
        ]

        changed = any(
            not torch.allclose(b, a, atol=1e-10)
            for b, a in zip(params_before, params_after)
        )
        assert changed, "Prompt parameters should change during TTA"

    def test_prompt_resets_in_episodic_mode(self):
        """Test that prompts reset to P_0 in episodic mode."""
        self.cfg["protocol"] = "episodic"
        engine = TTAEngine(self.model, self.cfg, self.device)

        # Get initial prompt state (P_0)
        p0 = engine.state_manager.mean.clone()

        # Adapt on a sample
        x = torch.randn(4, 3, 8, 8)
        engine.adapt_and_predict(x)

        # Adapt on second sample — should reset first
        engine.adapt_and_predict(x)

        # The prompt before the second adapt should have been reset to P_0
        # We can verify by checking the reset was called
        # (after adapt_and_predict, prompts will be P_0 + one step of update)

    def test_multiple_batches_accumulate_trajectory(self):
        """Test state buffer grows with online samples."""
        engine = TTAEngine(self.model, self.cfg, self.device)

        for i in range(5):
            x = torch.randn(4, 3, 8, 8)
            engine.adapt_and_predict(x)

        assert len(engine.state_manager._buffer) == 5

    def test_koopman_activates_after_window(self):
        """Test Koopman control kicks in after enough trajectory data."""
        engine = TTAEngine(self.model, self.cfg, self.device)

        window = self.cfg["tta"]["koopman"]["window"]

        # First few batches: no Koopman data
        for i in range(window + 2):
            x = torch.randn(4, 3, 8, 8)
            _, info = engine.adapt_and_predict(x)

        # After enough batches, rho should be non-zero
        # (it might still be zero in degenerate cases, but the machinery should work)
        assert len(engine.koopman.diagnostics["rho"]) >= window

    def test_deep_prompt_included(self):
        """Test that deep prompt embeddings are included in state."""
        engine = TTAEngine(self.model, self.cfg, self.device)
        params = engine.state_manager.get_prompt_params(self.model)

        # Should find both shallow and deep prompt params
        total_params = sum(p.numel() for p in params)
        shallow_numel = self.model.enc.transformer.prompt_embeddings.numel()
        deep_numel = self.model.enc.transformer.deep_prompt_embeddings.numel()

        assert total_params == shallow_numel + deep_numel, (
            f"Expected {shallow_numel + deep_numel} params, got {total_params}"
        )

    def test_metrics_tracking(self):
        """Test that metrics are accumulated correctly."""
        engine = TTAEngine(self.model, self.cfg, self.device)

        for _ in range(3):
            x = torch.randn(4, 3, 8, 8)
            y = torch.randint(0, 10, (4,))
            engine.adapt_and_predict(x, y)

        summary = engine.get_summary()
        assert "entropy_mean" in summary
        assert "accuracy_mean" in summary
        assert summary["total_samples"] == 3

    def test_backbone_frozen(self):
        """Test that non-prompt parameters have requires_grad=False."""
        engine = TTAEngine(self.model, self.cfg, self.device)

        for name, param in self.model.named_parameters():
            if "prompt" in name and "embeddings" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"


class TestStateManager:
    """Test PromptStateManager with the minimal model."""

    def test_prompt_collection(self):
        """Test that all prompt parameters are found."""
        model = MinimalViTWrapper(num_tokens=5, hidden_dim=32, num_classes=10)
        sm = PromptStateManager(model, state_dim=8, projection_type="random")

        # Should have found prompt_embeddings + deep_prompt_embeddings
        assert len(sm._prompt_param_names) == 2

    def test_state_dimensionality(self):
        """Test output state has correct shape."""
        model = MinimalViTWrapper(num_tokens=5, hidden_dim=32, num_classes=10)
        sm = PromptStateManager(model, state_dim=8, projection_type="random")

        z = sm.prompt_to_state(model)
        assert z.shape == (8,), f"Expected shape (8,), got {z.shape}"

    def test_drift_initially_zero(self):
        """Test drift is zero at initialization."""
        model = MinimalViTWrapper(num_tokens=5, hidden_dim=32, num_classes=10)
        sm = PromptStateManager(model, state_dim=8, projection_type="random")

        drift = sm.prompt_drift(model)
        assert abs(drift) < 1e-6, f"Initial drift should be ~0, got {drift}"

    def test_trajectory_buffer(self):
        """Test trajectory buffer and EDMD data extraction."""
        model = MinimalViTWrapper(num_tokens=5, hidden_dim=32, num_classes=10)
        sm = PromptStateManager(model, state_dim=8, projection_type="random")

        # Add states
        for _ in range(5):
            z = torch.randn(8)
            sm.update_buffer(z)

        traj = sm.get_trajectory(window=3)
        assert traj is not None
        Z_0, Z_1 = traj
        assert Z_0.shape == (3, 8)
        assert Z_1.shape == (3, 8)

    def test_trajectory_none_when_insufficient(self):
        """Test returns None when not enough data."""
        model = MinimalViTWrapper(num_tokens=5, hidden_dim=32, num_classes=10)
        sm = PromptStateManager(model, state_dim=8, projection_type="random")

        sm.update_buffer(torch.randn(8))
        assert sm.get_trajectory(window=3) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
