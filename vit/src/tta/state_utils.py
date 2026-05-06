"""
State utilities for TTA.

Handles prompt parameter → low-dimensional state projection,
supporting both shallow and deep prompts.

State definition:
    z_t = U^T vec(P_t) ∈ R^r

The projection U can be updated online from gradient history so that the
state space aligns with the *actual dynamics* of prompt adaptation,
enabling meaningful Gramian / HSV estimation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class PromptStateManager:
    """Manages prompt→state projection and state trajectory buffering.

    Attributes:
        r: State dimensionality.
        projection: Projection matrix U of shape (d, r).
        mean: Mean of prompt parameters for centering (from P_0).
        projection_is_dynamic: True after U has been rebuilt from gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        state_dim: int = 16,
        projection_type: str = "pca",
        device: Optional[torch.device] = None,
        adaptive_r: bool = False,
        max_state_dim: int = 512,
    ):
        self.adaptive_r = adaptive_r
        self.max_state_dim = max_state_dim
        self.r = state_dim
        self.projection_type = projection_type
        self.device = device or next(model.parameters()).device

        self._prompt_param_names = []
        self._prompt_param_shapes = []
        p0_vec = self._collect_prompt_vec(model)
        self.d = p0_vec.shape[0]
        self.mean = p0_vec.clone()

        if self.r > self.d:
            self.r = self.d

        self.projection = self._build_projection(p0_vec).to(self.device)
        self.projection_is_dynamic = False

        self._buffer: List[torch.Tensor] = []
        self._max_buffer = 200

    @staticmethod
    def _find_prompt_params(model: nn.Module):
        """Discover prompt parameters (shared by all param-access methods).

        Supports VPT (prompt_embeddings / deep_prompt_embeddings) and
        CLIP+CoOp (prompt_learner.ctx).

        Returns:
            List of (name, Parameter) tuples.
        """
        params = []
        enc = model.enc if hasattr(model, "enc") else model
        transformer = enc.transformer if hasattr(enc, "transformer") else enc
        for name, param in transformer.named_parameters():
            if "prompt" in name and "embeddings" in name:
                params.append((name, param))
        if not params and hasattr(model, "prompt_learner"):
            pl = model.prompt_learner
            if hasattr(pl, "ctx") and pl.ctx is not None:
                params.append(("prompt_learner.ctx", pl.ctx))
        return params

    def _collect_prompt_vec(self, model: nn.Module) -> torch.Tensor:
        """Flatten all prompt parameters into a single vector."""
        prompt_info = self._find_prompt_params(model)
        if not prompt_info:
            raise ValueError(
                "No prompt parameters found. Ensure model has either "
                "prompt_embeddings (VPT) or prompt_learner.ctx (CoOp)."
            )
        for name, param in prompt_info:
            self._prompt_param_names.append(name)
            self._prompt_param_shapes.append(param.shape)
        return torch.cat(
            [p.detach().reshape(-1) for _, p in prompt_info]
        ).to(self.device)

    def _build_projection(self, p0_vec: torch.Tensor) -> torch.Tensor:
        """Build *initial* projection matrix U (d, r).

        Before any gradient data is available, U is a random orthogonal
        matrix — semantically meaningless but dimensionally correct.
        Once enough gradient history is collected, call
        ``rebuild_projection_from_gradients`` to replace it with a
        dynamics-aligned basis.
        """
        U = torch.randn(self.d, self.r)
        U, _ = torch.linalg.qr(U)
        return U.to(self.device)

    # ---- Dynamic projection rebuild ----------------------------------------

    def rebuild_projection_from_gradients(
        self,
        grad_history: List[torch.Tensor],
        energy_threshold: float = 0.99,
    ) -> bool:
        """Replace U with the top-k PCA directions of the gradient history.

        When adaptive_r is True, k is determined purely by SVD energy and
        self.r is updated to match (the derivation's principled approach).
        When adaptive_r is False, k is clamped to the fixed self.r.

        Returns:
            True if rebuild succeeded, False otherwise.
        """
        G = torch.stack(grad_history)                       # (T, d)
        valid = torch.isfinite(G).all(dim=1)
        G = G[valid]
        if len(G) < max(min(self.r, 4), 4):
            return False

        G_np = G.cpu().numpy()
        try:
            _, s, Vt = np.linalg.svd(G_np, full_matrices=False)
        except np.linalg.LinAlgError:
            return False

        s_sq = s ** 2
        total = max(s_sq.sum(), 1e-12)
        cum = np.cumsum(s_sq) / total
        k_energy = int(np.searchsorted(cum, energy_threshold)) + 1

        if self.adaptive_r:
            k = max(2, min(k_energy, len(s), self.max_state_dim))
            self.r = k
        else:
            k = min(max(k_energy, self.r), self.r)
            k = min(k, len(s))

        V_new = Vt[:k].T                                   # (d, k)
        if k < self.r:
            pad = np.random.randn(self.d, self.r - k).astype(V_new.dtype)
            pad, _ = np.linalg.qr(pad)
            V_new = np.concatenate([V_new, pad[:, : self.r - k]], axis=1)

        U_cpu = torch.from_numpy(V_new).to(dtype=torch.float32)
        U_cpu, _ = torch.linalg.qr(U_cpu)                  # CPU to avoid cusolver
        self.projection = U_cpu.to(self.device)             # (d, k or r)
        self.projection_is_dynamic = True

        self._reproject_buffer()
        return True

    def _reproject_buffer(self):
        """After U changes, re-center the trajectory buffer.

        We cannot recover original p_vec from old z (lossy projection),
        so simply flush the buffer.  The Koopman / EDMD window will refill
        in a few steps.
        """
        self._buffer = []

    def prompt_to_state(self, model: nn.Module) -> torch.Tensor:
        """Extract current state z_t = U^T (vec(P_t) - vec(P_0)).

        Returns:
            z: State vector of shape (r,).
        """
        p_vec = self._get_current_prompt_vec(model)
        centered = p_vec - self.mean
        z = self.projection.T @ centered
        return z

    def _get_current_prompt_vec(self, model: nn.Module) -> torch.Tensor:
        """Get current prompt parameters as a flat vector."""
        params = self._find_prompt_params(model)
        return torch.cat(
            [p.detach().reshape(-1) for _, p in params]
        ).to(self.device)

    def update_buffer(self, z: torch.Tensor):
        """Append state to trajectory buffer."""
        self._buffer.append(z.detach().clone())
        if len(self._buffer) > self._max_buffer:
            self._buffer.pop(0)

    def get_trajectory(self, window: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get (Z_0, Z_1) trajectory matrices for EDMD.

        Args:
            window: Sliding window size W.

        Returns:
            (Z_0, Z_1) each of shape (W, r), or None if not enough data.
        """
        if len(self._buffer) < window + 1:
            return None

        recent = self._buffer[-(window + 1):]
        Z_0 = torch.stack(recent[:-1])  # (W, r)
        Z_1 = torch.stack(recent[1:])   # (W, r)
        return Z_0, Z_1

    def get_trajectory_adaptive(
        self, max_window: int, min_window: int = 2,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get (Z_0, Z_1) using all available data up to max_window.

        Unlike get_trajectory which requires exactly window+1 points,
        this starts returning data as soon as min_window+1 points are
        buffered, enabling earlier Koopman activation for per-step C1.
        """
        available = len(self._buffer) - 1
        if available < min_window:
            return None
        window = min(available, max_window)
        recent = self._buffer[-(window + 1):]
        Z_0 = torch.stack(recent[:-1])
        Z_1 = torch.stack(recent[1:])
        return Z_0, Z_1

    def get_prompt_params(self, model: nn.Module) -> List[nn.Parameter]:
        """Get list of prompt parameter references for gradient computation."""
        return [p for _, p in self._find_prompt_params(model)]

    def prompt_drift(self, model: nn.Module) -> float:
        """Compute ||P_t - P_0|| (prompt drift from initialization)."""
        p_vec = self._get_current_prompt_vec(model)
        return (p_vec - self.mean).norm().item()

    def reset_buffer(self):
        """Clear the state trajectory buffer (for episodic reset)."""
        self._buffer = []
