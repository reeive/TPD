"""§2.4 Hankel Update Router (decomposition + routing to d-space)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HURDecomposition:
    q: torch.Tensor  # explainability  (r,)
    p: torch.Tensor  # persistence     (r,)
    persistent: torch.Tensor  # s_t  (r,)
    oscillatory: torch.Tensor  # c_t  (r,)
    noise: torch.Tensor  # n_t  (r,)


@dataclass
class RoutedUpdate:
    routed: torch.Tensor  # final update in d-space
    q_mean: float
    p_mean: float


class HankelUpdateRouter:
    def __init__(
        self,
        beta_c: float = 0.25,
        beta_n: float = 0.1,
        kappa: float = 2.0,
        eps: float = 1e-6,
        mode: str = "hankel",
    ):
        self.beta_c = float(beta_c)
        self.beta_n = float(beta_n)
        self.kappa = float(kappa)
        self.eps = float(eps)
        self.mode = (mode or "hankel").lower()
        if self.mode not in ("hankel", "plain"):
            raise ValueError(
                f"Unknown HUR mode: {mode!r} (use hankel, plain)"
            )

    def decompose(
        self,
        y_curr: torch.Tensor,
        y_prev: torch.Tensor,
        a: torch.Tensor,
    ) -> HURDecomposition:
        """Per-mode TPD decomposition (prediction-informed, observation-directed)."""
        y_hat = a * y_prev
        r = y_curr - y_hat
        q = y_hat.square() / (y_hat.square() + r.square() + self.eps)
        p = 0.5 * (1.0 + torch.tanh(self.kappa * a))

        return HURDecomposition(
            q=q,
            p=p,
            persistent=q * p * y_curr,
            oscillatory=q * (1.0 - p) * y_curr,
            noise=(1.0 - q) * y_curr,
        )

    def route(
        self, decomp: HURDecomposition, Q: torch.Tensor, e: torch.Tensor
    ) -> RoutedUpdate:
        """Map decomposed components back to d-space and apply routing."""
        if self.mode == "plain":
            raise ValueError("plain mode: use route_plain(Q, y_curr, e) instead")
        S = Q @ decomp.persistent
        C = Q @ decomp.oscillatory
        N = Q @ decomp.noise + e

        routed = S + self.beta_c * C + self.beta_n * N
        return RoutedUpdate(
            routed=routed,
            q_mean=float(decomp.q.mean().item()),
            p_mean=float(decomp.p.mean().item()),
        )

    def route_plain(
        self, Q: torch.Tensor, y_curr: torch.Tensor, e: torch.Tensor
    ) -> RoutedUpdate:
        """Subspace back-projection only (ablation, no q/p or S/C/N)."""
        routed = Q @ y_curr + e
        return RoutedUpdate(
            routed=routed,
            q_mean=0.0,
            p_mean=0.0,
        )
