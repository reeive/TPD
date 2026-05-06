"""
C1: Koopman Spectral Controller.

Implements sliding-window EDMD to fit Koopman operator A_t, compute spectral
radius ρ_t, and provide spectral trust-region step-size control, early
stopping, and rollback decisions.

Supports fitting modes:
  - "full": Standard ridge-regularized EDMD (A is r×r dense).
  - "diagonal": Per-dimension growth-rate estimation (A = diag(a_1,...,a_r)).
  - "auto": Use full EDMD for ρ when W ≥ 3r and Gram is well-conditioned;
    otherwise diagonal. Diagonal a_i is always fitted for C2 (q/p decomposition).
    Each a_i is estimated from its own 1-D time series via least-squares,
    making the estimation well-conditioned regardless of state_dim.
    This allows high-dimensional state spaces (r=100-200) without
    underdetermination.

Also monitors von Neumann entropy S_vN and effective dimension d_eff
(inspired by ISK, ICLR 2026) as diagnostic signals for spectral diversity
analysis, without intervening in the EDMD fitting or step-size computation.

Theory (Proposition 1):
    If ρ(A) > 1, perturbations grow exponentially → collapse is inevitable.
    Using η_t = η_0 / max(1, ρ_t) compresses the effective spectral radius
    to ≤ 1, eliminating collapse.
"""

import math
import torch
from typing import Optional, Tuple, Dict, List


class KoopmanController:
    """Koopman-Spectral stability controller for TTA updates.

    Monitors the spectral radius of the Koopman operator fitted via
    sliding-window EDMD and controls:
      - Step size (spectral trust-region)
      - Early stopping
      - Prompt rollback

    Additionally monitors von Neumann entropy and effective spectral
    dimension as diagnostics (ISK-inspired, not used for control).
    """

    def __init__(
        self,
        base_lr: float = 1e-3,
        window_size: int = 10,
        ridge_lambda: float = 0.01,
        rollback_patience: int = 3,
        state_dim: int = 16,
        min_lr_ratio: float = 0.1,
        max_lr_ratio: float = 2.0,
        rho_threshold: float = 1.0,
        c1_mode: str = "full",
        device: Optional[torch.device] = None,
        auto_cond_threshold: float = 1e8,
    ):
        """
        Args:
            base_lr: Base learning rate η_0.
            window_size: Sliding window size W for EDMD.
            ridge_lambda: Ridge regularization λ for EDMD.
            rollback_patience: Number m of consecutive ρ_t > threshold steps
                before triggering rollback.
            state_dim: State dimensionality r (for sanity checks).
            min_lr_ratio: Minimum lr as fraction of η_0 (clamp floor).
            max_lr_ratio: Maximum lr as fraction of η_0 (clamp ceiling).
            rho_threshold: Threshold for spectral trust-region activation.
                η = η₀/max(threshold, ρ). Default 1.0 (original). Use 0.9
                for single-view mode (softer, earlier intervention).
            c1_mode: "full", "diagonal", or "auto" (see module docstring).
            device: Torch device.
            auto_cond_threshold: For c1_mode="auto", max allowed cond(Z_0^T Z_0)
                to use full EDMD for ρ (else diagonal ρ).
        """
        self.eta_0 = base_lr
        self.W = window_size
        self.ridge_lambda = ridge_lambda
        self.rollback_patience = rollback_patience
        self.r = state_dim
        self.rho_threshold = rho_threshold
        self.min_lr = base_lr * min_lr_ratio
        self.max_lr = base_lr * max_lr_ratio
        self.c1_mode = c1_mode
        self.auto_cond_threshold = auto_cond_threshold
        self.device = device or torch.device("cpu")

        # Internal state
        self._rho_history: List[float] = []
        self._consecutive_unstable = 0
        self._last_A_hat: Optional[torch.Tensor] = None
        self._last_diag_a: Optional[torch.Tensor] = None

        # Diagnostics
        self.diagnostics: Dict[str, list] = {
            "rho": [],
            "eta": [],
            "stable_ratio": [],
            "rollback_triggered": [],
            "vn_entropy": [],
            "d_eff": [],
            "unstable_dims": [],
        }

    # ------------------------------------------------------------------
    # ISK-inspired diagnostic utilities (monitoring only, not for control)
    # ------------------------------------------------------------------

    @staticmethod
    def von_neumann_entropy(eigenvalues: torch.Tensor) -> float:
        """Compute von Neumann entropy of the Koopman spectral weights.

        S_vN = -Σ_i w_i log(w_i),  where w_i = |λ_i|² / Σ_j |λ_j|²

        High S_vN → spectrally diverse (healthy).
        Low S_vN  → spectral collapse (degenerate).

        This is a diagnostic metric inspired by ISK (ICLR 2026) and is NOT
        used for control decisions — only for monitoring and visualization.

        Args:
            eigenvalues: Complex eigenvalues, shape (r,).

        Returns:
            S_vN: von Neumann entropy (scalar).
        """
        magnitudes_sq = eigenvalues.abs().pow(2)
        total = magnitudes_sq.sum()
        if total < 1e-12:
            return 0.0
        weights = magnitudes_sq / total
        weights = weights.clamp(min=1e-12)
        entropy = -(weights * weights.log()).sum().item()
        return entropy

    @staticmethod
    def effective_dimension(vn_entropy: float) -> float:
        """Compute effective dimension from von Neumann entropy.

        d_eff = exp(S_vN)

        Ranges from 1 (fully collapsed) to r (fully uniform).

        Args:
            vn_entropy: von Neumann entropy S_vN.

        Returns:
            d_eff: Effective spectral dimension.
        """
        return math.exp(vn_entropy)

    # ------------------------------------------------------------------
    # Core EDMD and spectral analysis
    # ------------------------------------------------------------------

    def fit_koopman(
        self, Z_0: torch.Tensor, Z_1: torch.Tensor
    ) -> torch.Tensor:
        """Fit Koopman operator via ridge-regularized EDMD.

        A_hat = (Z_0^T Z_0 + λI)^{-1} Z_0^T Z_1

        Args:
            Z_0: Previous states, shape (W, r).
            Z_1: Next states, shape (W, r).

        Returns:
            A_hat: Fitted Koopman operator, shape (r, r).
        """
        Z_0 = Z_0.to(self.device)
        Z_1 = Z_1.to(self.device)

        gram = Z_0.T @ Z_0 + self.ridge_lambda * torch.eye(
            Z_0.shape[1], device=self.device
        )
        rhs = Z_0.T @ Z_1

        # Move to CPU for linalg.solve to avoid MAGMA errors on small matrices
        try:
            A_hat = torch.linalg.solve(gram.cpu(), rhs.cpu()).to(self.device)
        except (RuntimeError, ValueError):
            # Fallback: pseudo-inverse (more numerically stable)
            try:
                gram_inv = torch.linalg.pinv(gram.cpu())
                A_hat = (gram_inv @ rhs.cpu()).to(self.device)
            except (RuntimeError, ValueError):
                # Last resort: identity (no adaptation)
                A_hat = torch.eye(Z_0.shape[1], device=self.device)
        return A_hat

    def fit_koopman_diagonal(
        self, Z_0: torch.Tensor, Z_1: torch.Tensor
    ) -> torch.Tensor:
        """Fit diagonal Koopman operator: z_{i,t+1} ≈ a_i * z_{i,t}.

        Each dimension is estimated independently via scalar least-squares:
            a_i = Σ_t z_{i,t} * z_{i,t+1} / (Σ_t z_{i,t}² + λ)

        This is always well-conditioned: W data points for 1 parameter,
        allowing state_dim to be arbitrarily high (100-200) without
        underdetermination.

        Args:
            Z_0: Previous states, shape (W, r).
            Z_1: Next states, shape (W, r).

        Returns:
            diag_a: Per-dimension growth rates, shape (r,).
                    The effective A_hat = diag(diag_a).
        """
        Z_0 = Z_0.to(self.device)
        Z_1 = Z_1.to(self.device)

        numerator = (Z_0 * Z_1).sum(dim=0)       # (r,) — Σ z_{i,t}·z_{i,t+1}
        denominator = (Z_0 * Z_0).sum(dim=0)     # (r,) — Σ z_{i,t}²
        denominator = denominator + self.ridge_lambda

        diag_a = numerator / denominator          # (r,)

        # Clamp extreme values for numerical stability
        diag_a = diag_a.clamp(-10.0, 10.0)
        self._last_diag_a = diag_a.detach()
        return diag_a

    def compute_spectral_stats_diagonal(
        self, diag_a: torch.Tensor
    ) -> Tuple[float, float, float, float, int]:
        """Compute spectral statistics from diagonal Koopman coefficients.

        Args:
            diag_a: Per-dimension growth rates, shape (r,).

        Returns:
            rho: Spectral radius max_i |a_i|.
            stable_ratio: Fraction of dimensions with |a_i| < 1.
            vn_entropy: von Neumann entropy of |a_i|² weights.
            d_eff: Effective spectral dimension.
            n_unstable: Number of dimensions with |a_i| > threshold.
        """
        magnitudes = diag_a.abs()
        valid = torch.isfinite(magnitudes)
        if valid.sum() == 0:
            return 1.0, 0.5, math.log(max(self.r, 1)), float(self.r), 0

        magnitudes = magnitudes[valid]
        rho = magnitudes.max().item()
        stable_ratio = (magnitudes < 1.0).float().mean().item()
        n_unstable = int((magnitudes > self.rho_threshold).sum().item())

        # von Neumann entropy on |a_i|² weights
        mag_sq = magnitudes.pow(2)
        total = mag_sq.sum()
        if total < 1e-12:
            vn_ent = 0.0
        else:
            weights = mag_sq / total
            weights = weights.clamp(min=1e-12)
            vn_ent = -(weights * weights.log()).sum().item()
        d_eff = math.exp(vn_ent)

        return rho, stable_ratio, vn_ent, d_eff, n_unstable

    def compute_spectral_stats(
        self, A_hat: torch.Tensor
    ) -> Tuple[float, float, float, float]:
        """Compute spectral statistics of the Koopman operator.

        Args:
            A_hat: Koopman operator, shape (r, r).

        Returns:
            rho: Spectral radius max_i |λ_i(A)|.
            stable_ratio: Fraction of eigenvalues with |λ_i| < 1.
            vn_entropy: von Neumann entropy (ISK diagnostic).
            d_eff: Effective spectral dimension (ISK diagnostic).
        """
        try:
            eigenvalues = torch.linalg.eigvals(A_hat.cpu())
        except (RuntimeError, ValueError):
            # Eigendecomposition failed — report neutral values
            return 1.0, 0.5, math.log(max(self.r, 1)), float(self.r)

        magnitudes = eigenvalues.abs()
        # Filter out any NaN/Inf eigenvalues
        valid = torch.isfinite(magnitudes)
        if valid.sum() == 0:
            return 1.0, 0.5, math.log(max(self.r, 1)), float(self.r)
        magnitudes = magnitudes[valid]

        rho = magnitudes.max().item()
        stable_ratio = (magnitudes < 1.0).float().mean().item()

        # ISK diagnostics (monitoring only)
        vn_ent = self.von_neumann_entropy(eigenvalues[valid])
        d_eff = self.effective_dimension(vn_ent)

        return rho, stable_ratio, vn_ent, d_eff

    def _should_use_full_edmd(self, Z_0: torch.Tensor) -> bool:
        """True if window is sufficiently overdetermined and Gram is stable."""
        Z_0 = Z_0.to(self.device)
        w, r = Z_0.shape[0], Z_0.shape[1]
        if w < 3 * r:
            return False
        gram = Z_0.T @ Z_0 + self.ridge_lambda * torch.eye(
            r, device=self.device)
        try:
            c = torch.linalg.cond(gram.cpu()).item()
        except (RuntimeError, ValueError):
            return False
        if not math.isfinite(c) or c > self.auto_cond_threshold:
            return False
        return True

    # ------------------------------------------------------------------
    # Step-size control and rollback
    # ------------------------------------------------------------------

    def get_adapted_lr(
        self,
        Z_0: Optional[torch.Tensor] = None,
        Z_1: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict]:
        """Compute adapted learning rate via spectral trust-region.

        η_t = η_0 / max(1, ρ_t)

        If not enough data for EDMD, returns base lr.

        Args:
            Z_0: Previous states, shape (W, r), or None.
            Z_1: Next states, shape (W, r), or None.

        Returns:
            eta_t: Adapted learning rate.
            info: Diagnostic info dict.
        """
        info = {
            "rho": None,
            "stable_ratio": None,
            "rollback": False,
            "early_stop": False,
            "vn_entropy": None,
            "d_eff": None,
            "n_unstable": 0,
            "diag_a": None,
            "c1_rho_source": None,
        }

        # Not enough data yet — use base lr
        if Z_0 is None or Z_1 is None:
            eta_t = self.eta_0
            info["rho"] = 0.0
            info["stable_ratio"] = 1.0
            info["vn_entropy"] = math.log(max(self.r, 1))
            info["d_eff"] = float(self.r)
            self._last_diag_a = None
            self._record_diagnostics(eta_t, info)
            return eta_t, info

        # Always fit diagonal a_i for C2 (q/p decomposition) and stability fallback
        diag_a = self.fit_koopman_diagonal(Z_0, Z_1)
        info["diag_a"] = diag_a.detach().clone()

        use_full_for_rho = False
        if self.c1_mode == "full":
            use_full_for_rho = True
        elif self.c1_mode == "auto":
            use_full_for_rho = self._should_use_full_edmd(Z_0)

        if use_full_for_rho:
            A_hat = self.fit_koopman(Z_0, Z_1)
            self._last_A_hat = A_hat
            rho, stable_ratio, vn_ent, d_eff = self.compute_spectral_stats(
                A_hat)
            info["c1_rho_source"] = "full"
        else:
            rho, stable_ratio, vn_ent, d_eff, n_unstable = \
                self.compute_spectral_stats_diagonal(diag_a)
            info["n_unstable"] = n_unstable
            info["c1_rho_source"] = "diagonal"
        info["rho"] = rho
        info["stable_ratio"] = stable_ratio
        info["vn_entropy"] = vn_ent
        info["d_eff"] = d_eff

        # Spectral trust-region step size:
        #   ρ < threshold → η = η₀ (no intervention)
        #   ρ ≥ threshold → η = η₀ × threshold/ρ (reduce proportionally)
        if rho >= self.rho_threshold:
            eta_t = self.eta_0 * self.rho_threshold / rho
        else:
            eta_t = self.eta_0
        # Clamp to safe range
        eta_t = max(self.min_lr, min(eta_t, self.max_lr))

        # Track instability for rollback
        if rho > self.rho_threshold:
            self._consecutive_unstable += 1
        else:
            self._consecutive_unstable = 0

        # Rollback decision
        if self._consecutive_unstable >= self.rollback_patience:
            info["rollback"] = True
            self._consecutive_unstable = 0

        # Early stop signal: if spectral radius is rising fast
        if len(self._rho_history) >= 3:
            recent = self._rho_history[-3:]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                if rho > recent[-1]:
                    info["early_stop"] = True

        self._rho_history.append(rho)
        self._record_diagnostics(eta_t, info)

        return eta_t, info

    def _record_diagnostics(self, eta_t: float, info: Dict):
        """Record diagnostics for later analysis."""
        self.diagnostics["rho"].append(info.get("rho", 0))
        self.diagnostics["eta"].append(eta_t)
        self.diagnostics["stable_ratio"].append(info.get("stable_ratio", 1))
        self.diagnostics["rollback_triggered"].append(info.get("rollback", False))
        self.diagnostics["vn_entropy"].append(info.get("vn_entropy", 0))
        self.diagnostics["d_eff"].append(info.get("d_eff", 0))
        self.diagnostics["unstable_dims"].append(info.get("n_unstable", 0))

    # ------------------------------------------------------------------
    # GAR-based step-size control (Gradient Consistency Monitor)
    # ------------------------------------------------------------------

    def __init_gar_state(self):
        """Lazily initialize GAR tracking state."""
        if not hasattr(self, '_gar_history'):
            self._gar_history: List[float] = []
            self._gar_ema: float = 1.0
            self._gar_ema_alpha: float = 0.9
            if not hasattr(self, '_gar_target'):
                self._gar_target: float = 0.5
            self._gar_low_count: int = 0
            self._gar_rollback_patience: int = 5

    def get_lr_from_gar(
        self, gar: float,
        gar_target: Optional[float] = None,
    ) -> Tuple[float, Dict]:
        """Compute adapted learning rate from Gradient Agreement Ratio.

        GAR = σ₁² / Σσᵢ² measures how much of the gradient energy is in the
        dominant (agreed-upon) direction.  High GAR → confident update;
        low GAR → cautious/skip.

        This replaces the EDMD-based spectral radius control.  The Koopman
        connection: GAR is an observable of the adaptation system, and its
        temporal evolution reveals stability (stable/declining GAR).

        Args:
            gar: Gradient Agreement Ratio ∈ [1/G, 1].
            gar_target: Override for the target GAR (default from init).

        Returns:
            eta_t: Adapted learning rate.
            info: Diagnostic dict.
        """
        self.__init_gar_state()

        if gar_target is None:
            gar_target = self._gar_target

        info = {
            "gar": gar,
            "gar_ema": self._gar_ema,
            "gar_trend": 0.0,
            "rho": gar,
            "rollback": False,
            "early_stop": False,
        }

        self._gar_history.append(gar)
        self._gar_ema = self._gar_ema_alpha * self._gar_ema + (
            1 - self._gar_ema_alpha) * gar

        alpha_t = min(1.0, gar / max(gar_target, 1e-6))
        eta_t = self.eta_0 * alpha_t
        eta_t = max(self.min_lr, min(eta_t, self.max_lr))

        info["gar_ema"] = self._gar_ema

        if len(self._gar_history) >= 5:
            recent = self._gar_history[-5:]
            trend = recent[-1] - recent[0]
            info["gar_trend"] = trend

        if gar < gar_target * 0.5:
            self._gar_low_count += 1
        else:
            self._gar_low_count = 0

        if self._gar_low_count >= self._gar_rollback_patience:
            info["rollback"] = True
            self._gar_low_count = 0

        if len(self._gar_history) >= 4:
            recent = self._gar_history[-4:]
            if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                if gar < gar_target * 0.7:
                    info["early_stop"] = True

        info["eta"] = eta_t
        self.diagnostics["rho"].append(gar)
        self.diagnostics["eta"].append(eta_t)
        self.diagnostics["rollback_triggered"].append(info["rollback"])

        return eta_t, info

    def reset(self):
        """Reset internal state (for episodic mode)."""
        self._rho_history = []
        self._consecutive_unstable = 0
        self._last_A_hat = None
        self._last_diag_a = None
        if hasattr(self, '_gar_history'):
            self._gar_history = []
            self._gar_ema = 1.0
            self._gar_low_count = 0
