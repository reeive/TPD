"""
C2: Hankel-Balanced Update Operator (HBUO).

Three operating modes:
  c2_mode='vsga' — View Sub-batch Gradient Agreement (recommended).
      Split views into G groups, compute per-group gradients, SVD of the
      gradient matrix identifies the effective (agreed-upon) subspace.
      The mean gradient is projected onto this subspace for denoising.
      Returns GAR (Gradient Agreement Ratio) for C1 step-size control.

  c2_mode='fast' — PCA on gradient history (lightweight, no Gramians).
  c2_mode='full' — Dynamics-aligned HBUO with Gramians.

C2-fast variants (c2_variant attribute):
  'centered'   — PCA on centered gradients
  'uncentered' — SVD on raw history (default, better conditioning)
  'gated'      — anomaly-gated: only filters when cos(g,g_proj) < threshold
  'shrink'     — soft projection keeping partial out-of-subspace component
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List


class HBUO:
    """Hankel-Balanced Update Operator for effective gradient filtering."""

    def __init__(
        self,
        state_dim: int = 16,
        num_perturbations: int = 5,
        perturbation_scale: float = 0.01,
        cross_episode: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.r = state_dim
        self.M = num_perturbations
        self.delta_scale = perturbation_scale
        self.cross_episode = cross_episode
        self.device = device or torch.device("cpu")

        self._W_c: Optional[torch.Tensor] = None
        self._W_o: Optional[torch.Tensor] = None
        self._H: Optional[torch.Tensor] = None
        self._V: Optional[torch.Tensor] = None
        self._hsv: Optional[torch.Tensor] = None
        self._tau: Optional[float] = None
        self._W_c_sqrt: Optional[torch.Tensor] = None

        self.diagnostics: Dict[str, list] = {
            "hsv_spectrum": [],
            "tau": [],
            "effective_ratio": [],
        }

        self._grad_history: List[torch.Tensor] = []
        self._max_grad_history = 200
        self._warmup_steps = 3
        self._call_count = 0
        self._k_eff_cap = state_dim

        self.c2_variant = "uncentered"
        self.c2_gate_threshold = 0.85
        self.c2_gate_blend = 0.5

        # Exposed for diagnostics: last SVD spectrum and effective rank
        self.last_svd_spectrum: Optional[np.ndarray] = None
        self.last_k_eff: int = 0
        self.c2_shrink_ratio = 0.5

        self._projection_rebuilt = False
        self._rebuild_threshold = max(state_dim + 4, 20)

    @torch.no_grad()
    def estimate_controllability_gramian(self, model: nn.Module, x: torch.Tensor, state_manager) -> torch.Tensor:
        """Estimate W_c from prompt perturbations (no forward pass needed)."""
        z_base = state_manager.prompt_to_state(model)
        prompt_params = state_manager.get_prompt_params(model)
        delta_z_list = []

        for _ in range(self.M):
            originals = [p.data.clone() for p in prompt_params]
            for p in prompt_params:
                p.data.add_(torch.randn_like(p) * self.delta_scale)
            z_perturbed = state_manager.prompt_to_state(model)
            delta_z_list.append(z_perturbed - z_base)
            for p, orig in zip(prompt_params, originals):
                p.data.copy_(orig)

        delta_Z = torch.stack(delta_z_list)
        W_c = (delta_Z.T @ delta_Z) / self.M
        W_c = W_c + 1e-6 * torch.eye(self.r, device=self.device)
        self._W_c = W_c
        return W_c

    def estimate_observability_gramian(self, model: nn.Module, x: torch.Tensor, state_manager) -> torch.Tensor:
        prompt_params = state_manager.get_prompt_params(model)
        x_sub = x[:min(4, x.shape[0])]

        with torch.no_grad():
            base_logits = model(x_sub)
            base_logits_mean = base_logits.mean(dim=0)

        num_classes = base_logits_mean.shape[0]
        J = torch.zeros(num_classes, self.r, device=self.device)
        eps = self.delta_scale
        U = state_manager.projection

        for i in range(self.r):
            originals = [p.data.clone() for p in prompt_params]
            delta_p_flat = eps * U[:, i]
            offset = 0
            for p in prompt_params:
                numel = p.numel()
                p.data.add_(delta_p_flat[offset:offset + numel].reshape(p.shape))
                offset += numel

            with torch.no_grad():
                perturbed_logits = model(x_sub)
                perturbed_mean = perturbed_logits.mean(dim=0)

            J[:, i] = (perturbed_mean - base_logits_mean) / eps
            for p, orig in zip(prompt_params, originals):
                p.data.copy_(orig)

        W_o = J.T @ J
        W_o = W_o + 1e-6 * torch.eye(self.r, device=self.device)
        self._W_o = W_o
        return W_o

    def compute_hankel(
        self,
        W_c: Optional[torch.Tensor] = None,
        W_o: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if W_c is None:
            W_c = self._W_c
        if W_o is None:
            W_o = self._W_o
        if W_c is None or W_o is None:
            raise ValueError("Gramians not estimated yet. Call estimate_*_gramian first.")

        W_c_cpu = W_c.cpu()
        W_o_cpu = W_o.cpu()
        eigvals_c, eigvecs_c = torch.linalg.eigh(W_c_cpu)
        eigvals_c = eigvals_c.clamp(min=1e-8)
        W_c_sqrt = eigvecs_c @ torch.diag(eigvals_c.sqrt()) @ eigvecs_c.T
        H = W_c_sqrt @ W_o_cpu @ W_c_sqrt

        try:
            eigvals_h, eigvecs_h = torch.linalg.eigh(H)
        except (RuntimeError, ValueError):
            self._H = H.to(self.device)
            self._V = torch.eye(self.r, device=self.device)
            self._hsv = torch.ones(self.r, device=self.device)
            self._tau = 1.0
            self._W_c_sqrt = W_c_sqrt.to(self.device)
            return self._H, self._hsv, self._tau

        eigvals_h = eigvals_h.clamp(min=0)
        hsv = eigvals_h.sqrt()
        tau = eigvals_h.mean().item()

        idx = torch.argsort(hsv, descending=True)
        hsv = hsv[idx]
        eigvecs_h = eigvecs_h[:, idx]

        self._H = H.to(self.device)
        self._V = eigvecs_h.to(self.device)
        self._hsv = hsv.to(self.device)
        self._tau = tau
        self._W_c_sqrt = W_c_sqrt.to(self.device)
        self.diagnostics["hsv_spectrum"].append(hsv.detach().cpu().tolist())
        self.diagnostics["tau"].append(tau)
        self.diagnostics["effective_ratio"].append(
            (eigvals_h > tau).float().mean().item())
        return H, hsv, tau

    def filter_gradient(self, g_z: torch.Tensor, energy_threshold: float = 0.95) -> torch.Tensor:
        """Filter state-space gradient by projecting onto high-HSV subspace."""
        if self._hsv is None or self._V is None:
            return g_z

        hsv = self._hsv
        V = self._V

        hsv_sq = hsv.pow(2)
        total = hsv_sq.sum().item()
        if total < 1e-12:
            return g_z

        cum_energy = torch.cumsum(hsv_sq, dim=0) / total
        k_eff = int((cum_energy < energy_threshold).sum().item()) + 1
        k_eff = max(1, min(k_eff, len(hsv)))

        V_k = V[:, :k_eff]
        return V_k @ (V_k.T @ g_z)

    filter = filter_gradient

    def has_hsv(self) -> bool:
        return self._hsv is not None and self._V is not None

    def filter_gradient_fast(self, g: torch.Tensor, energy_threshold: float = 0.95) -> torch.Tensor:
        """C2-fast: PCA-based gradient subspace filtering.

        c2_variant controls the projection strategy:
          'centered'   — original: PCA on centered history (poor conditioning)
          'uncentered' — SVD on raw history (default, better conditioning)
          'gated'      — only activates when gradient deviates from subspace
          'shrink'     — soft projection keeping partial out-of-subspace signal
        """
        self._call_count += 1
        self._grad_history.append(g.detach().cpu().clone())
        if len(self._grad_history) > self._max_grad_history:
            self._grad_history.pop(0)

        if self._call_count <= self._warmup_steps:
            return g
        if len(self._grad_history) < 3:
            return g

        variant = self.c2_variant
        try:
            G = torch.stack(self._grad_history)
            valid = torch.isfinite(G).all(dim=1)
            G = G[valid]
            if len(G) < 2:
                return g

            if variant == "centered":
                G_for_svd = (G - G.mean(dim=0)).cpu().numpy()
            else:
                G_for_svd = G.cpu().numpy()

            _, s, Vt = np.linalg.svd(G_for_svd, full_matrices=False)
            self.last_svd_spectrum = s.copy()

            s_sq = s ** 2
            total = max(s_sq.sum(), 1e-12)
            cum_energy = np.cumsum(s_sq) / total
            k_eff = int(np.searchsorted(cum_energy, energy_threshold)) + 1
            k_eff = max(1, min(k_eff, self._k_eff_cap, len(s)))
            self.last_k_eff = k_eff

            V_eff = Vt[:k_eff].T
            g_np = g.cpu().numpy()
            if not np.isfinite(g_np).all():
                return g
            g_proj = V_eff @ (V_eff.T @ g_np)

            if variant == "gated":
                g_norm = np.linalg.norm(g_np)
                g_proj_norm = np.linalg.norm(g_proj)
                if g_norm < 1e-12 or g_proj_norm < 1e-12:
                    return g
                cos_sim = np.dot(g_np, g_proj) / (g_norm * g_proj_norm)
                if cos_sim >= self.c2_gate_threshold:
                    return torch.from_numpy(g_proj).to(dtype=g.dtype, device=g.device)
                bl = self.c2_gate_blend
                g_blended = (1 - bl) * g_proj + bl * g_np
                return torch.from_numpy(g_blended).to(dtype=g.dtype, device=g.device)

            if variant == "shrink":
                g_out = g_np - g_proj
                g_filtered = g_proj + self.c2_shrink_ratio * g_out
                return torch.from_numpy(g_filtered).to(dtype=g.dtype, device=g.device)

            return torch.from_numpy(g_proj).to(dtype=g.dtype, device=g.device)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            return g

    # ------------------------------------------------------------------
    # VSGA: View Sub-batch Gradient Agreement
    # ------------------------------------------------------------------

    def filter_gradient_vsga(
        self,
        group_grads: List[torch.Tensor],
        energy_threshold: float = 0.8,
    ) -> Tuple[torch.Tensor, float, Dict]:
        """Denoise gradient via inter-group agreement (SVD of gradient matrix).

        Given G per-group gradients, identifies the effective subspace where
        groups agree and projects the mean gradient onto it.  This is a
        practical Hankel analysis: large singular values = jointly
        controllable+observable directions (signal); small = noise.

        Args:
            group_grads: List of G gradient vectors, each shape (d,).
            energy_threshold: Fraction of SVD energy to retain.

        Returns:
            g_filtered: Denoised gradient, shape (d,).
            gar: Gradient Agreement Ratio (σ₁² / Σσᵢ²).
            info: Diagnostic dict with SVD spectrum, k_eff, etc.
        """
        G = len(group_grads)
        info = {"svd_spectrum": [], "k_eff": 0, "gar": 0.0}

        if G < 2:
            g = group_grads[0]
            info["gar"] = 1.0
            return g, 1.0, info

        g_mean = torch.stack(group_grads).mean(dim=0)

        valid_grads = []
        for g in group_grads:
            if torch.isfinite(g).all():
                valid_grads.append(g)
        if len(valid_grads) < 2:
            info["gar"] = 1.0
            return g_mean, 1.0, info

        try:
            G_mat = torch.stack(valid_grads)  # (G, d)
            G_np = G_mat.cpu().float().numpy()

            _, s, Vt = np.linalg.svd(G_np, full_matrices=False)

            s_sq = s ** 2
            total_energy = max(s_sq.sum(), 1e-12)
            gar = float(s_sq[0] / total_energy)

            cum_energy = np.cumsum(s_sq) / total_energy
            k_eff = int(np.searchsorted(cum_energy, energy_threshold)) + 1
            k_eff = max(1, min(k_eff, len(s)))

            info["svd_spectrum"] = s.tolist()
            info["k_eff"] = k_eff
            info["gar"] = gar
            self.last_svd_spectrum = s.copy()
            self.last_k_eff = k_eff

            V_k = Vt[:k_eff].T  # (d, k_eff)
            g_np = g_mean.cpu().float().numpy()
            if not np.isfinite(g_np).all():
                return g_mean, gar, info

            g_proj = V_k @ (V_k.T @ g_np)
            g_filtered = torch.from_numpy(g_proj).to(
                dtype=g_mean.dtype, device=g_mean.device)

            self._grad_history.append(g_filtered.detach().cpu().clone())
            if len(self._grad_history) > self._max_grad_history:
                self._grad_history.pop(0)

            return g_filtered, gar, info

        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            info["gar"] = 1.0 / G
            return g_mean, 1.0 / G, info

    def maybe_rebuild_projection(self, state_manager) -> bool:
        """Rebuild state_manager.projection from gradient history.

        Called by the engine when c2_mode='full'.  Returns True if the
        projection was (re)built this call, meaning Gramians should be
        re-estimated on the next forward pass.
        """
        if self._projection_rebuilt:
            return False
        if len(self._grad_history) < self._rebuild_threshold:
            return False
        ok = state_manager.rebuild_projection_from_gradients(
            self._grad_history, energy_threshold=0.99)
        if ok:
            self._projection_rebuilt = True
            self._invalidate_gramians()
        return ok

    def _invalidate_gramians(self):
        """Clear cached Gramians so they get re-estimated on new basis."""
        self._W_c = None
        self._W_o = None
        self._H = None
        self._V = None
        self._hsv = None
        self._tau = None
        self._W_c_sqrt = None

    def update_gramians(self, model: nn.Module, x: torch.Tensor, state_manager) -> Tuple[torch.Tensor, torch.Tensor, float]:
        W_c = self.estimate_controllability_gramian(model, x, state_manager)
        W_o = self.estimate_observability_gramian(model, x, state_manager)
        return self.compute_hankel(W_c, W_o)

    def reset(self):
        """Reset state for new episode."""
        if not self.cross_episode:
            self._W_c = None
            self._W_o = None
            self._H = None
            self._V = None
            self._hsv = None
            self._tau = None
            self._W_c_sqrt = None
            self._grad_history = []
            self._call_count = 0
            self._projection_rebuilt = False
