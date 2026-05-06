"""§2.3 Adaptive Koopman Control (diagonal DMD or full operator, spectral radius, rollback)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class AKCResult:
    coefficients: torch.Tensor  # original a_{1,t}, ..., a_{r,t}  (r,)
    stabilized: torch.Tensor  # ā_i with |ā_i| ≤ ρ̄ (per mode) or scalar-damped
    rho: float
    scale: float  # min(1, ρ̄/ρ_t)  (for diagonal: may vary per mode; we expose scalar in result)
    rollback: bool
    stable: bool


class AdaptiveKoopmanControl:
    def __init__(
        self,
        rho_threshold: float = 1.0,
        patience: int = 3,
        min_scale: float = 0.1,
        eps: float = 1e-8,
        mode: str = "diagonal",
    ):
        self.rho_threshold = float(rho_threshold)
        self.patience = int(patience)
        self.min_scale = float(min_scale)
        self.eps = float(eps)
        self.mode = (mode or "diagonal").lower()
        if self.mode not in ("diagonal", "full_koopman"):
            raise ValueError(
                f"Unknown AKC mode: {mode!r} (use diagonal, full_koopman)"
            )
        self._consecutive_unstable = 0

    @staticmethod
    def diagonal_dmd_a(Y: torch.Tensor, eps: float = 1e-8) -> Optional[torch.Tensor]:
        """Unstabilized diagonal DMD gains (a_i), for HUR blocks without full AKC."""
        if Y is None or Y.shape[0] < 2:
            return None
        Y_prev = Y[:-1].T  # r × (W-1)
        Y_next = Y[1:].T
        a = (Y_prev * Y_next).sum(dim=1) / Y_prev.square().sum(dim=1).clamp_min(eps)
        return a

    def estimate(self, Y: Optional[torch.Tensor]) -> Optional[AKCResult]:
        """Y: (W, r) projected history; returns None if insufficient data."""
        if Y is None or Y.shape[0] < 2:
            return None

        if self.mode == "diagonal":
            return self._estimate_diagonal(Y)
        return self._estimate_full_koopman(Y)

    def _estimate_diagonal(self, Y: torch.Tensor) -> AKCResult:
        a = self.diagonal_dmd_a(Y, self.eps)
        assert a is not None
        rho = float(a.abs().max().item())

        clamp_ratio = torch.clamp(
            self.rho_threshold / a.abs().clamp_min(self.eps), max=1.0
        )
        a_bar = a * clamp_ratio

        stable = rho <= self.rho_threshold
        if stable:
            self._consecutive_unstable = 0
            scale = 1.0
        else:
            self._consecutive_unstable += 1
            scale = max(self.min_scale, self.rho_threshold / max(rho, self.eps))

        rollback = (not stable) and self._consecutive_unstable >= self.patience
        return AKCResult(
            coefficients=a,
            stabilized=a_bar,
            rho=rho,
            scale=scale,
            rollback=rollback,
            stable=stable,
        )

    def _estimate_full_koopman(self, Y: torch.Tensor) -> AKCResult:
        """A = Y_next @ pinv(Y_prev), ρ = max |λ| (global spectral radius)."""
        Y_prev = Y[:-1].T  # r × m
        Y_next = Y[1:].T
        m = Y_prev.shape[1]
        if m < 1:
            return None
        A = Y_next @ torch.linalg.pinv(Y_prev)  # r × r, real
        A = A.to(dtype=Y_prev.dtype)
        if not torch.isfinite(A).all():
            return self._estimate_diagonal(Y)

        eig = torch.linalg.eigvals(A)
        if eig.is_complex() or A.is_complex():
            rho = float(torch.max(torch.abs(eig)).item())
        else:
            rho = float(torch.max(torch.abs(eig)).item())

        coefficients = torch.diagonal(A, 0)
        c_scalar = min(1.0, self.rho_threshold / max(rho, self.eps))
        a_bar = coefficients * c_scalar

        stable = rho <= self.rho_threshold
        if stable:
            self._consecutive_unstable = 0
            scale = 1.0
        else:
            self._consecutive_unstable += 1
            scale = max(self.min_scale, self.rho_threshold / max(rho, self.eps))

        rollback = (not stable) and self._consecutive_unstable >= self.patience
        return AKCResult(
            coefficients=coefficients,
            stabilized=a_bar,
            rho=rho,
            scale=scale,
            rollback=rollback,
            stable=stable,
        )

    def state_dict(self) -> Dict:
        return {"consecutive_unstable": self._consecutive_unstable}

    def load_state_dict(self, sd: Dict) -> None:
        self._consecutive_unstable = int(sd.get("consecutive_unstable", 0))

    def reset(self) -> None:
        self._consecutive_unstable = 0
