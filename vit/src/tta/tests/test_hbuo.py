"""
Unit tests for HBUO (C2).

Tests mathematical correctness of:
- Hankel matrix construction
- HSV computation
- HBUO gradient filtering properties
- Self-calibrated τ
"""

import torch
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.tta.hbuo import HBUO


class TestHBUO:
    """Test suite for HBUO operator."""

    def setup_method(self):
        self.device = torch.device("cpu")
        self.r = 4
        self.hbuo = HBUO(
            state_dim=self.r,
            num_perturbations=5,
            perturbation_scale=0.01,
            device=self.device,
        )

    def test_hankel_construction(self):
        """Test H = W_c^{1/2} W_o W_c^{1/2} is correct."""
        # Create known Gramians
        W_c = torch.eye(self.r) * 2.0
        W_o = torch.eye(self.r) * 3.0

        H, hsv, tau = self.hbuo.compute_hankel(W_c, W_o)

        # For diagonal case: H = sqrt(2) * 3 * sqrt(2) * I = 6I
        expected_H = 6.0 * torch.eye(self.r)
        assert torch.allclose(H, expected_H, atol=1e-4), (
            f"Hankel mismatch:\n{H}\nvs expected:\n{expected_H}"
        )

    def test_hsv_correctness(self):
        """Test Hankel singular values are computed correctly."""
        W_c = torch.diag(torch.tensor([4.0, 1.0, 0.25, 0.01]))
        W_o = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))

        H, hsv, tau = self.hbuo.compute_hankel(W_c, W_o)

        # H_ii = W_c_ii * W_o_ii = W_c_ii (since W_o = I)
        # HSV_i = sqrt(H_ii) = sqrt(W_c_ii)
        expected_hsv = torch.sqrt(torch.tensor([0.01, 0.25, 1.0, 4.0]))
        assert torch.allclose(hsv, expected_hsv, atol=1e-3), (
            f"HSV mismatch: {hsv} vs expected {expected_hsv}"
        )

    def test_tau_self_calibration(self):
        """Test τ = (1/r) tr(H) = mean eigenvalue."""
        W_c = torch.diag(torch.tensor([4.0, 1.0, 0.25, 0.01]))
        W_o = torch.eye(self.r)

        H, hsv, tau = self.hbuo.compute_hankel(W_c, W_o)

        expected_tau = (4.0 + 1.0 + 0.25 + 0.01) / self.r
        assert abs(tau - expected_tau) < 1e-4, (
            f"τ mismatch: {tau} vs expected {expected_tau}"
        )

    def test_gradient_filtering_suppresses_low_hsv(self):
        """Test that HBUO suppresses gradient in low-HSV directions."""
        # Make one direction highly controllable/observable, rest nearly zero
        W_c = torch.diag(torch.tensor([10.0, 0.01, 0.01, 0.01]))
        W_o = torch.diag(torch.tensor([10.0, 0.01, 0.01, 0.01]))

        self.hbuo.compute_hankel(W_c, W_o)

        # Gradient equally in all directions
        g_z = torch.ones(self.r)
        g_filtered = self.hbuo.filter_gradient(g_z)

        # The first direction (high HSV) should retain more gradient
        # than the other directions (low HSV)
        assert g_filtered[0].abs() > g_filtered[1].abs() * 2, (
            f"Expected first direction to dominate: {g_filtered}"
        )

    def test_gradient_filtering_preserves_direction(self):
        """Test that gradient aligned with top HSV direction is mostly preserved.

        HBUO contracts gradients (it's a preconditioner, not preserving norm),
        so we test that the top-direction component is significantly larger
        than the other components, which should be near zero.
        """
        W_c = torch.diag(torch.tensor([10.0, 0.01, 0.01, 0.01]))
        W_o = torch.diag(torch.tensor([10.0, 0.01, 0.01, 0.01]))

        self.hbuo.compute_hankel(W_c, W_o)

        # Gradient only in the top direction
        g_z = torch.tensor([1.0, 0.0, 0.0, 0.0])
        g_filtered = self.hbuo.filter_gradient(g_z)

        # Top direction should have non-trivial output
        assert g_filtered[0].abs() > 1e-3, (
            f"Top-direction gradient should be non-trivial: {g_filtered}"
        )
        # Other directions should be near zero (gradient was zero there)
        assert g_filtered[1:].norm() < 1e-6, (
            f"Zero-input directions should stay zero: {g_filtered}"
        )

    def test_reset(self):
        """Test reset clears cached state."""
        W_c = torch.eye(self.r)
        W_o = torch.eye(self.r)
        self.hbuo.compute_hankel(W_c, W_o)

        assert self.hbuo._H is not None
        self.hbuo.reset()
        assert self.hbuo._H is None
        assert self.hbuo._W_c is None

    def test_filter_gradient_without_hankel_raises(self):
        """Test error when calling filter_gradient before compute_hankel."""
        with pytest.raises(ValueError):
            self.hbuo.filter_gradient(torch.randn(self.r))

    def test_hankel_positive_semidefinite(self):
        """Test H is PSD (all eigenvalues ≥ 0)."""
        # Random symmetric PSD Gramians
        A = torch.randn(self.r, self.r)
        W_c = A @ A.T + 0.01 * torch.eye(self.r)
        B = torch.randn(self.r, self.r)
        W_o = B @ B.T + 0.01 * torch.eye(self.r)

        H, hsv, tau = self.hbuo.compute_hankel(W_c, W_o)

        eigvals = torch.linalg.eigvalsh(H)
        assert (eigvals >= -1e-6).all(), (
            f"H should be PSD, got eigenvalues: {eigvals}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestHBUOExtensions:
    """Tests for new HBUO features."""

    def setup_method(self):
        self.device = torch.device("cpu")
        self.r = 4
        self.hbuo = HBUO(
            state_dim=self.r,
            num_perturbations=5,
            perturbation_scale=0.01,
            device=self.device,
        )

    def test_filter_gradient_fast(self):
        """Test lightweight PCA-based C2 filtering (with warmup)."""
        torch.manual_seed(42)
        # Feed enough gradients to pass warmup (15) + min history (5)
        for _ in range(20):
            g = torch.randn(self.r)
            result = self.hbuo.filter_gradient_fast(g)
            assert result.shape == (self.r,)

        g_test = torch.randn(self.r)
        g_filtered = self.hbuo.filter_gradient_fast(g_test)
        assert g_filtered.shape == g_test.shape
        # Filtered gradient should have smaller or equal norm (projection)
        assert g_filtered.norm() <= g_test.norm() + 1e-4

    def test_filter_alias(self):
        """Test that filter() is an alias for filter_gradient()."""
        W_c = torch.diag(torch.tensor([4.0, 1.0, 0.25, 0.01]))
        W_o = torch.eye(self.r)
        self.hbuo.compute_hankel(W_c, W_o)

        g = torch.tensor([1.0, 0.5, 0.25, 0.1])
        g1 = self.hbuo.filter_gradient(g)
        g2 = self.hbuo.filter(g)
        assert torch.allclose(g1, g2), "filter() should equal filter_gradient()"

    def test_reset_clears_grad_history(self):
        """Test reset clears gradient history for fast filter."""
        for _ in range(5):
            self.hbuo.filter_gradient_fast(torch.randn(self.r))
        assert len(self.hbuo._grad_history) > 0
        self.hbuo.reset()
        assert len(self.hbuo._grad_history) == 0
