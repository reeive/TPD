"""
PEFT-Native HBUO: Coefficient-Space Hankel-Balanced Update Operator.

Operates in a low-rank prompt coefficient space (r_p << Nd) rather than the
full gradient space. The key steps:

  1. Build a structure-aware prompt basis B_t from gradient history PCA.
  2. Project gradients into coefficient space: q_t = B_t^T @ g_t.
  3. Construct block-Hankel states from recent coefficients.
  4. Fit a lifted linear input-output model with predictive outputs
     (future loss reduction, rho improvement, drift penalty).
  5. Compute balanced truncation to get an oblique projector that retains
     modes which are simultaneously controllable and observable.
  6. Filter in coefficient space, reconstruct prompt-only update.

Falls back to HAUO-fast (gradient history PCA) during warmup or when
insufficient data is available for balanced truncation.
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple


class PEFTNativeHBUO:
    """Coefficient-space Hankel-balanced gradient filter for prompt TTA."""

    def __init__(
        self,
        prompt_basis_dim: int = 16,
        hankel_len: int = 5,
        horizon: int = 3,
        shrink_ratio: float = 0.5,
        energy_threshold: float = 0.95,
        update_freq: int = 10,
        warmup: int = 30,
        gram_horizon: int = 20,
        max_history: int = 200,
        cross_episode: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.r_p = prompt_basis_dim
        self.L = hankel_len
        self.H = horizon
        self.alpha = shrink_ratio
        self.energy_threshold = energy_threshold
        self.update_freq = update_freq
        self.warmup = warmup
        self.gram_horizon = gram_horizon
        self.max_history = max_history
        self.cross_episode = cross_episode
        self.device = device or torch.device("cpu")

        self.B_t: Optional[torch.Tensor] = None

        self._grad_history: List[torch.Tensor] = []
        self._coeff_history: List[torch.Tensor] = []
        self._loss_history: List[float] = []
        self._rho_history: List[float] = []
        self._drift_history: List[float] = []

        self.Pi_bal: Optional[torch.Tensor] = None

        self._call_count = 0
        self._warmup_steps = 3
        self._last_basis_step = -999
        self._last_balance_step = -999

        self.last_svd_spectrum: Optional[np.ndarray] = None
        self.last_k_eff: int = 0
        self.last_hsv: Optional[np.ndarray] = None
        self.last_bal_rank: int = 0
        self.bal_active: bool = False

    def _build_prompt_basis(self):
        """PCA of gradient history -> orthonormal basis B_t in R^{Nd x r_p}."""
        n = len(self._grad_history)
        if n < max(3, self.r_p):
            return
        G = torch.stack(self._grad_history[-self.max_history:])
        try:
            G_np = G.cpu().float().numpy()
            _, s, Vt = np.linalg.svd(G_np, full_matrices=False)
            k = min(self.r_p, len(s), Vt.shape[0])
            self.B_t = torch.from_numpy(Vt[:k].T.copy()).to(
                dtype=torch.float32, device=self.device)
            self.last_svd_spectrum = s.copy()
            self._last_basis_step = self._call_count
        except (np.linalg.LinAlgError, ValueError):
            pass

    def _fit_and_balance(self):
        """Fit lifted linear model and compute balanced truncation projector."""
        n_coeffs = len(self._coeff_history)
        n_outputs = len(self._loss_history)
        needed = self.L + self.H + 5
        if n_coeffs < needed or n_outputs < needed:
            return

        actual_rp = self._coeff_history[0].shape[0]
        hankel_dim = self.L * actual_rp

        X_list, X_next_list, Y_list = [], [], []
        max_t = min(n_coeffs - self.H, n_outputs - self.H)
        for t in range(self.L, max_t - 1):
            coeffs = self._coeff_history[t - self.L:t]
            x_t = torch.cat(coeffs)
            coeffs_next = self._coeff_history[t - self.L + 1:t + 1]
            x_next = torch.cat(coeffs_next)
            if t + self.H < n_outputs:
                dl = self._loss_history[t + self.H] - self._loss_history[t]
                dr = self._rho_history[t + self.H] - self._rho_history[t]
                dd = -(self._drift_history[t + self.H] - self._drift_history[t])
                y_t = torch.tensor([dl, dr, dd],
                                   dtype=torch.float32, device=self.device)
                X_list.append(x_t)
                X_next_list.append(x_next)
                Y_list.append(y_t)

        n_data = len(X_list)
        if n_data < max(10, hankel_dim // 2):
            return

        try:
            X = torch.stack(X_list).to(self.device)
            X_next = torch.stack(X_next_list).to(self.device)
            Y = torch.stack(Y_list).to(self.device)

            lam = 0.01
            I_d = torch.eye(hankel_dim, device=self.device)
            XtX_reg = X.T @ X + lam * I_d
            A_p = torch.linalg.solve(XtX_reg, X.T @ X_next).T
            C_p = torch.linalg.solve(XtX_reg, X.T @ Y).T

            K = min(self.gram_horizon, n_data // 2, 30)
            W_c = torch.zeros(hankel_dim, hankel_dim, device=self.device)
            W_o = torch.zeros(hankel_dim, hankel_dim, device=self.device)
            Ak = torch.eye(hankel_dim, device=self.device)
            for k in range(K):
                W_c += Ak @ Ak.T
                W_o += Ak.T @ C_p.T @ C_p @ Ak
                Ak = A_p @ Ak
                if Ak.abs().max() > 1e6:
                    break

            W_c += 1e-6 * I_d
            W_o += 1e-6 * I_d

            WcWo = W_c @ W_o
            WcWo_sym = 0.5 * (WcWo + WcWo.T)
            eigvals, eigvecs = torch.linalg.eigh(WcWo_sym)
            hsv = eigvals.clamp(min=0).sqrt()
            idx = hsv.argsort(descending=True)
            hsv = hsv[idx]
            eigvecs = eigvecs[:, idx]
            self.last_hsv = hsv.detach().cpu().numpy()

            total = hsv.sum().item()
            if total < 1e-12:
                return

            cumsum = hsv.cumsum(0)
            k_arr = (cumsum >= 0.9 * total).nonzero(as_tuple=True)[0]
            k_eff = max(1, k_arr[0].item() + 1) if len(k_arr) > 0 else hankel_dim
            k_eff = min(k_eff, actual_rp, hankel_dim)
            self.last_bal_rank = k_eff

            Phi_full = eigvecs[:, :k_eff]
            Phi_current = Phi_full[:actual_rp, :]
            if Phi_current.shape[1] > 0 and Phi_current.abs().max() > 1e-12:
                U_c, _, _ = torch.linalg.svd(Phi_current, full_matrices=False)
                k_use = min(k_eff, U_c.shape[1])
                self.Pi_bal = U_c[:, :k_use] @ U_c[:, :k_use].T
                self.bal_active = True
                self._last_balance_step = self._call_count
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            pass

    def filter_gradient(
        self,
        g: torch.Tensor,
        energy_threshold: Optional[float] = None,
        loss_t: Optional[float] = None,
        rho_t: Optional[float] = None,
        drift_t: Optional[float] = None,
    ) -> torch.Tensor:
        if energy_threshold is None:
            energy_threshold = self.energy_threshold

        self._call_count += 1
        self._grad_history.append(g.detach().clone())
        if len(self._grad_history) > self.max_history:
            self._grad_history.pop(0)

        if loss_t is not None:
            self._loss_history.append(float(loss_t))
            self._rho_history.append(float(rho_t) if rho_t is not None else 0.0)
            self._drift_history.append(float(drift_t) if drift_t is not None else 0.0)
            for h in (self._loss_history, self._rho_history, self._drift_history):
                if len(h) > self.max_history:
                    h.pop(0)

        if self._call_count < self.warmup:
            if (self._call_count - self._last_basis_step) >= self.update_freq:
                self._build_prompt_basis()
            return self._fallback_pca(g, energy_threshold)

        if (self._call_count - self._last_basis_step) >= self.update_freq:
            self._build_prompt_basis()

        if self.B_t is None:
            return self._fallback_pca(g, energy_threshold)

        q_t = self.B_t.T @ g
        self._coeff_history.append(q_t.detach().clone())
        if len(self._coeff_history) > self.max_history:
            self._coeff_history.pop(0)

        if (self._call_count - self._last_balance_step) >= self.update_freq:
            self._fit_and_balance()

        if self.bal_active and self.Pi_bal is not None:
            q_proj = self.Pi_bal @ q_t
            q_filtered = q_proj + self.alpha * (q_t - q_proj)
            g_filtered = self.B_t @ q_filtered
            g_basis = self.B_t @ q_t
            g_residual = g - g_basis
            return g_filtered + self.alpha * g_residual
        else:
            return self._fallback_pca(g, energy_threshold)

    def _fallback_pca(self, g: torch.Tensor, energy_threshold: float) -> torch.Tensor:
        if len(self._grad_history) < 3 or self._call_count <= self._warmup_steps:
            return g
        try:
            G = torch.stack(self._grad_history[-min(50, len(self._grad_history)):])
            valid = torch.isfinite(G).all(dim=1)
            G = G[valid]
            if len(G) < 2:
                return g
            G_np = G.cpu().float().numpy()
            _, s, Vt = np.linalg.svd(G_np, full_matrices=False)
            s_sq = s ** 2
            total = max(s_sq.sum(), 1e-12)
            cum = np.cumsum(s_sq) / total
            k = max(1, min(int(np.searchsorted(cum, energy_threshold)) + 1, len(s)))
            self.last_k_eff = k
            V_k = Vt[:k].T
            g_np = g.cpu().float().numpy()
            if not np.isfinite(g_np).all():
                return g
            g_proj = V_k @ (V_k.T @ g_np)
            g_out = g_np - g_proj
            g_filtered = g_proj + self.alpha * g_out
            return torch.from_numpy(g_filtered).to(dtype=g.dtype, device=g.device)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            return g

    def filter_gradient_vsga(
        self,
        group_grads: List[torch.Tensor],
        energy_threshold: Optional[float] = None,
        loss_t: Optional[float] = None,
        rho_t: Optional[float] = None,
        drift_t: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float, Dict]:
        if energy_threshold is None:
            energy_threshold = self.energy_threshold
        G = len(group_grads)
        info: Dict = {"svd_spectrum": [], "k_eff": 0, "gar": 0.0}
        if G < 2:
            gf = self.filter_gradient(group_grads[0], energy_threshold, loss_t, rho_t, drift_t)
            info["gar"] = 1.0
            return gf, 1.0, info

        g_mean = torch.stack(group_grads).mean(dim=0)
        valid = [g for g in group_grads if torch.isfinite(g).all()]
        if len(valid) < 2:
            gf = self.filter_gradient(g_mean, energy_threshold, loss_t, rho_t, drift_t)
            info["gar"] = 1.0
            return gf, 1.0, info

        try:
            G_mat = torch.stack(valid)
            G_np = G_mat.cpu().float().numpy()
            _, s, Vt = np.linalg.svd(G_np, full_matrices=False)
            s_sq = s ** 2
            total_e = max(s_sq.sum(), 1e-12)
            gar = float(s_sq[0] / total_e)
            cum = np.cumsum(s_sq) / total_e
            k_eff = max(1, min(int(np.searchsorted(cum, energy_threshold)) + 1, len(s)))
            info.update({"svd_spectrum": s.tolist(), "k_eff": k_eff, "gar": gar})
            V_k = Vt[:k_eff].T
            g_np = g_mean.cpu().float().numpy()
            if not np.isfinite(g_np).all():
                g_denoised = g_mean
            else:
                g_proj = V_k @ (V_k.T @ g_np)
                g_denoised = torch.from_numpy(g_proj).to(dtype=g_mean.dtype, device=g_mean.device)
            gf = self.filter_gradient(g_denoised, energy_threshold, loss_t, rho_t, drift_t)
            return gf, gar, info
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            gf = self.filter_gradient(g_mean, energy_threshold, loss_t, rho_t, drift_t)
            info["gar"] = 1.0 / G
            return gf, 1.0 / G, info

    def reset(self):
        self._coeff_history.clear()
        self._loss_history.clear()
        self._rho_history.clear()
        self._drift_history.clear()
        self.Pi_bal = None
        self.bal_active = False
        self._last_balance_step = -999
        if not self.cross_episode:
            self._grad_history.clear()
            self.B_t = None
            self._call_count = 0
            self._last_basis_step = -999

    def has_hsv(self) -> bool:
        return self.bal_active and self.Pi_bal is not None
