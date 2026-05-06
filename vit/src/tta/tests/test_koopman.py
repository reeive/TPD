"""
Unit tests for Koopman Spectral Controller (C1).

Tests:
- EDMD fitting and spectral recovery
- Spectral radius computation
- Spectral trust-region step-size adaptation
- Rollback logic
- ISK-inspired diagnostics (von Neumann entropy, effective dimension)
"""

import math
import torch
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.tta.koopman_controller import KoopmanController


class TestKoopmanController:
    """Test suite for core KoopmanController functionality."""

    def setup_method(self):
        self.device = torch.device("cpu")
        self.ctrl = KoopmanController(
            base_lr=0.01,
            window_size=5,
            ridge_lambda=0.01,
            rollback_patience=3,
            state_dim=4,
            device=self.device,
        )

    def test_fit_koopman_known_dynamics(self):
        """Test EDMD recovers a known linear system A.

        EDMD solves: Z_1 ≈ Z_0 A (row convention).
        """
        r = 4
        A_true = torch.eye(r)
        A_true[0, 1] = 0.05
        A_true[1, 0] = -0.05

        torch.manual_seed(42)
        z_list = [torch.randn(r)]
        for _ in range(50):
            z_list.append(z_list[-1] @ A_true + 0.1 * torch.randn(r))

        Z_0 = torch.stack(z_list[:-1])
        Z_1 = torch.stack(z_list[1:])

        A_hat = self.ctrl.fit_koopman(Z_0, Z_1)

        eig_true = torch.linalg.eigvals(A_true).abs().sort().values
        eig_hat = torch.linalg.eigvals(A_hat).abs().sort().values
        eig_error = (eig_hat - eig_true).norm() / eig_true.norm()
        assert eig_error < 0.3, f"Spectral recovery error too large: {eig_error:.4f}"

    def test_spectral_radius_stable(self):
        """Test spectral radius < 1 for a stable system."""
        A_stable = 0.8 * torch.eye(4)
        rho, stable_ratio, _, _ = self.ctrl.compute_spectral_stats(A_stable)
        assert rho < 1.0
        assert stable_ratio == 1.0

    def test_spectral_radius_unstable(self):
        """Test spectral radius > 1 for an unstable system."""
        A_unstable = 1.2 * torch.eye(4)
        rho, stable_ratio, _, _ = self.ctrl.compute_spectral_stats(A_unstable)
        assert rho > 1.0
        assert stable_ratio == 0.0

    def test_step_size_reduction(self):
        """Test η_t < η_0 when ρ > 1."""
        A_unstable = 1.5 * torch.eye(4)
        z_list = [torch.randn(4)]
        for _ in range(6):
            z_list.append(A_unstable @ z_list[-1] + 0.01 * torch.randn(4))

        Z_0 = torch.stack(z_list[:5])
        Z_1 = torch.stack(z_list[1:6])
        eta_t, info = self.ctrl.get_adapted_lr(Z_0, Z_1)

        assert eta_t < self.ctrl.eta_0
        assert info["rho"] > 1.0

    def test_step_size_unchanged_when_stable(self):
        """Test η_t = η_0 when ρ ≤ 1."""
        A_stable = 0.5 * torch.eye(4)
        z_list = [torch.randn(4)]
        for _ in range(6):
            z_list.append(A_stable @ z_list[-1] + 0.001 * torch.randn(4))

        Z_0 = torch.stack(z_list[:5])
        Z_1 = torch.stack(z_list[1:6])
        eta_t, _ = self.ctrl.get_adapted_lr(Z_0, Z_1)

        assert abs(eta_t - self.ctrl.eta_0) < 1e-6

    def test_rollback_triggered(self):
        """Test rollback after m consecutive unstable steps."""
        A_unstable = 1.5 * torch.eye(4)
        z_list = [torch.randn(4)]
        for _ in range(6):
            z_list.append(A_unstable @ z_list[-1] + 0.01 * torch.randn(4))

        Z_0 = torch.stack(z_list[:5])
        Z_1 = torch.stack(z_list[1:6])

        rollback_seen = False
        for _ in range(self.ctrl.rollback_patience + 1):
            _, info = self.ctrl.get_adapted_lr(Z_0, Z_1)
            if info["rollback"]:
                rollback_seen = True
                break

        assert rollback_seen, "Rollback should have been triggered"

    def test_no_data_returns_base_lr(self):
        """Test default behavior when no trajectory data available."""
        eta_t, info = self.ctrl.get_adapted_lr(None, None)
        assert eta_t == self.ctrl.eta_0
        assert info["rho"] == 0.0
        assert info["vn_entropy"] is not None
        assert info["d_eff"] is not None

    def test_reset(self):
        """Test reset clears state."""
        self.ctrl._rho_history = [1.0, 1.2, 1.5]
        self.ctrl._consecutive_unstable = 3
        self.ctrl.reset()
        assert len(self.ctrl._rho_history) == 0
        assert self.ctrl._consecutive_unstable == 0

    def test_lr_clamped_within_bounds(self):
        """Test eta is clamped within [min_lr, max_lr]."""
        ctrl = KoopmanController(
            base_lr=0.01, state_dim=4,
            min_lr_ratio=0.1, max_lr_ratio=2.0,
        )
        # Very unstable system → rho >> 1 → eta should be clamped at min_lr
        A_very_unstable = 100.0 * torch.eye(4)
        z_list = [torch.randn(4)]
        for _ in range(6):
            z_list.append(A_very_unstable @ z_list[-1] + 0.01 * torch.randn(4))
        Z_0 = torch.stack(z_list[:5])
        Z_1 = torch.stack(z_list[1:6])
        eta_t, _ = ctrl.get_adapted_lr(Z_0, Z_1)
        assert eta_t >= ctrl.min_lr - 1e-10, f"eta {eta_t} < min_lr {ctrl.min_lr}"
        assert eta_t <= ctrl.max_lr + 1e-10, f"eta {eta_t} > max_lr {ctrl.max_lr}"

    def test_nan_eigenvalue_fallback(self):
        """Test graceful handling when eigenvalues are NaN."""
        ctrl = KoopmanController(base_lr=0.01, state_dim=4)
        # Create a matrix with NaN entries
        A_nan = torch.full((4, 4), float('nan'))
        rho, sr, vn, deff = ctrl.compute_spectral_stats(A_nan)
        # Should return neutral values, not crash
        assert rho >= 0
        assert deff > 0


class TestISKDiagnostics:
    """Test suite for ISK-inspired diagnostic metrics (monitoring only)."""

    def test_von_neumann_entropy_uniform(self):
        """S_vN = log(r) for uniform spectral weights."""
        eigenvalues = torch.tensor([1.0 + 0j, 1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
        s_vn = KoopmanController.von_neumann_entropy(eigenvalues)
        assert abs(s_vn - math.log(4)) < 1e-4

    def test_von_neumann_entropy_collapsed(self):
        """S_vN → 0 when one eigenvalue dominates."""
        eigenvalues = torch.tensor([10.0 + 0j, 0.01 + 0j, 0.01 + 0j, 0.01 + 0j])
        s_vn = KoopmanController.von_neumann_entropy(eigenvalues)
        assert s_vn < 0.1

    def test_effective_dimension_bounds(self):
        """d_eff ranges from 1 (collapsed) to r (uniform)."""
        assert abs(KoopmanController.effective_dimension(math.log(4)) - 4.0) < 1e-4
        assert abs(KoopmanController.effective_dimension(0.0) - 1.0) < 1e-4

    def test_diagnostics_recorded(self):
        """ISK diagnostic fields are recorded in diagnostics dict."""
        ctrl = KoopmanController(state_dim=4, device=torch.device("cpu"))
        ctrl.get_adapted_lr(None, None)

        assert "vn_entropy" in ctrl.diagnostics
        assert "d_eff" in ctrl.diagnostics
        assert len(ctrl.diagnostics["vn_entropy"]) == 1
        assert len(ctrl.diagnostics["d_eff"]) == 1

    def test_spectral_stats_include_isk(self):
        """compute_spectral_stats returns vn_entropy and d_eff."""
        ctrl = KoopmanController(state_dim=4, device=torch.device("cpu"))
        A = 0.9 * torch.eye(4)
        rho, sr, vn, deff = ctrl.compute_spectral_stats(A)
        assert vn > 0  # should have positive entropy
        assert deff > 0  # should have positive effective dim
        assert abs(vn - math.log(4)) < 1e-4  # uniform → max entropy


class TestKoopmanSpectralTrustRegion:
    """Test the spectral trust-region mathematical property."""

    def test_effective_spectral_radius_bounded(self):
        """Verify η_t = η_0/max(1,ρ_t) → effective ρ̃ ≤ 1."""
        ctrl = KoopmanController(base_lr=0.01, state_dim=4)

        for rho_test in [0.5, 0.9, 1.0, 1.5, 2.0, 5.0]:
            eta_t = ctrl.eta_0 / max(1.0, rho_test)
            effective_rho = rho_test * (eta_t / ctrl.eta_0)
            assert effective_rho <= 1.0 + 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
