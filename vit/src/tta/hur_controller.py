from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HURSplit:
    y_hat: torch.Tensor
    residual: torch.Tensor
    explainability: torch.Tensor
    persistence: torch.Tensor
    persistent: torch.Tensor
    oscillatory: torch.Tensor
    noise: torch.Tensor


@dataclass
class RoutedUpdate:
    persistent: torch.Tensor
    oscillatory: torch.Tensor
    residual: torch.Tensor
    routed: torch.Tensor


class HankelUpdateRouter:
    """Diagonal HUR router derived from previous-step Koopman coefficients.

    Routing modes (controlled by `routing_mode`):
        "fixed"       — original: global β_c, β_n constants.
        "adaptive_bn" — β_n adapts to mean explainability q̄:
                         β_n_eff = bn_max·(1-q̄) + bn_min·q̄
                         Low q → more pass-through; high q → more suppression.
        "per_mode_bc" — β_c per mode based on |a_i|:
                         β_c_i = β_c · σ(bc_tau · (1 - |a_i|))
                         Amplifying modes (|a_i|>1) → stronger suppression.
    """

    def __init__(
        self,
        beta_c: float = 0.25,
        beta_n: float = 0.1,
        kappa: float = 2.0,
        eps: float = 1e-6,
        routing_mode: str = "fixed",
        bn_max: float = 0.5,
        bn_min: float = 0.05,
        bc_tau: float = 5.0,
    ):
        self.beta_c = float(beta_c)
        self.beta_n = float(beta_n)
        self.kappa = float(kappa)
        self.eps = float(eps)
        self.routing_mode = routing_mode
        self.bn_max = float(bn_max)
        self.bn_min = float(bn_min)
        self.bc_tau = float(bc_tau)

    def split(
        self,
        current_projected: torch.Tensor,
        previous_projected: torch.Tensor,
        diagonal_coefficients: torch.Tensor,
    ) -> HURSplit:
        y_hat = diagonal_coefficients * previous_projected
        residual = current_projected - y_hat

        explainability = y_hat.square() / (
            y_hat.square() + residual.square() + self.eps
        )
        persistence = 0.5 * (1.0 + torch.tanh(self.kappa * diagonal_coefficients))

        persistent = explainability * persistence * current_projected
        oscillatory = explainability * (1.0 - persistence) * current_projected
        noise = (1.0 - explainability) * current_projected

        return HURSplit(
            y_hat=y_hat,
            residual=residual,
            explainability=explainability,
            persistence=persistence,
            persistent=persistent,
            oscillatory=oscillatory,
            noise=noise,
        )

    def route(
        self,
        split: HURSplit,
        basis: torch.Tensor,
        out_of_subspace_residual: Optional[torch.Tensor] = None,
        diagonal_coefficients: Optional[torch.Tensor] = None,
    ) -> RoutedUpdate:
        persistent = basis @ split.persistent

        # --- oscillatory routing ---
        if self.routing_mode == "per_mode_bc" and diagonal_coefficients is not None:
            beta_c_vec = self.beta_c * torch.sigmoid(
                self.bc_tau * (1.0 - diagonal_coefficients.abs())
            )
            oscillatory = basis @ (beta_c_vec * split.oscillatory)
        else:
            oscillatory = self.beta_c * (basis @ split.oscillatory)

        # --- residual routing ---
        residual = basis @ split.noise
        if out_of_subspace_residual is not None:
            residual = residual + out_of_subspace_residual

        if self.routing_mode == "adaptive_bn":
            q_bar = float(split.explainability.mean().item())
            beta_n_eff = self.bn_max * (1.0 - q_bar) + self.bn_min * q_bar
        else:
            beta_n_eff = self.beta_n

        routed = persistent + oscillatory + beta_n_eff * residual
        return RoutedUpdate(
            persistent=persistent,
            oscillatory=oscillatory,
            residual=residual,
            routed=routed,
        )
