"""
Koopman Trajectory Multi-View (KTMV).

Generates dynamics-informed virtual prompt views using Koopman eigenmodes,
replacing image-augmentation-based multi-view (TPT) with a dynamics-based
approach that is natively compatible with online TTA.

Core idea:
    Instead of augmenting images, perturb the prompt state along the
    principal dynamical modes (Koopman eigenvectors) of the adaptation
    trajectory.  The resulting virtual views produce diverse predictions
    that feed a marginal-entropy loss—exactly the mechanism of TPT—but
    without requiring batch_size=1 or per-image augmentation.

Theory:
    The same Koopman operator Â that C1 uses for spectral-radius control
    is eigen-decomposed: Â = V Λ V⁻¹.  Each column v_m of V is a mode
    direction in the reduced state space R^r.  Perturbations along v_m
    are "dynamically plausible" because they follow the system's own
    characteristic directions.

    KTMV loss:
      L = H( 1/(M+1) [ softmax(f(x; P)) + Σ_m softmax(f(x; P + UΔz_m)) ] )

    where Δz_m = sign_m · α · ‖z_t‖ · v_m / ‖v_m‖

Gradient flow:
    Only logits from the original prompt P have gradient; virtual-view
    logits are detached.  The marginal entropy still provides an improved
    signal: predictions sensitive to Koopman perturbations receive larger
    gradients, while robust predictions receive smaller ones.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple


class KoopmanTrajectoryMultiView:
    """Generate virtual prompt views from Koopman eigenmodes.

    Attributes:
        M: Number of virtual views (excluding the original).
        scale: Base perturbation magnitude relative to ‖z_t‖.
    """

    def __init__(
        self,
        n_views: int = 4,
        perturbation_scale: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        self.M = n_views
        self.scale = perturbation_scale
        self.device = device or torch.device("cpu")

        self._cached_modes: Optional[torch.Tensor] = None
        self._cache_counter = 0
        self._cache_interval = 5

    def generate_prompt_perturbations(
        self,
        K: torch.Tensor,
        z_t: torch.Tensor,
        projection_U: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Generate M perturbation vectors in full prompt space.

        Args:
            K: Koopman operator of shape (r, r).
            z_t: Current reduced state of shape (r,).
            projection_U: Projection matrix U of shape (d, r).

        Returns:
            List of M prompt-space perturbation vectors, each of shape (d,).
        """
        r = K.shape[0]
        z_scale = z_t.norm().item()
        if z_scale < 1e-10:
            return []

        modes = self._get_modes(K, r)
        if modes is None:
            return []

        perturbations: List[torch.Tensor] = []
        for m in range(self.M):
            mode_idx = m % r
            sign = 1.0 if (m // r) % 2 == 0 else -1.0

            direction = modes[:, mode_idx]
            d_norm = direction.norm()
            if d_norm < 1e-12:
                continue
            direction = direction / d_norm

            delta_z = sign * self.scale * z_scale * direction
            delta_p = projection_U @ delta_z
            perturbations.append(delta_p)

        return perturbations

    def _get_modes(self, K: torch.Tensor, r: int) -> Optional[torch.Tensor]:
        """Extract real Koopman eigenmodes, cached for efficiency."""
        self._cache_counter += 1
        if (self._cached_modes is not None
                and self._cache_counter % self._cache_interval != 0):
            return self._cached_modes

        try:
            eigenvalues, eigenvectors = torch.linalg.eig(K.cpu())
        except (RuntimeError, ValueError):
            return self._cached_modes

        V = eigenvectors.real.to(self.device)
        magnitudes = eigenvalues.abs()
        idx = torch.argsort(magnitudes, descending=True)
        V = V[:, idx]

        self._cached_modes = V
        return V

    @staticmethod
    def compute_marginal_entropy(
        logits_views: List[torch.Tensor],
    ) -> torch.Tensor:
        """Per-image marginal entropy across KTMV prompt views.

        The first element in logits_views retains its gradient; the rest
        are expected to be detached by the caller.

        Args:
            logits_views: List of (B, C) logit tensors.  logits_views[0]
                has gradient; the others are detached.

        Returns:
            Scalar entropy loss.
        """
        probs_list = [F.softmax(l, dim=-1) for l in logits_views]
        avg_probs = torch.stack(probs_list).mean(dim=0)
        entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum(dim=-1)
        return entropy.mean()

    def reset(self):
        """Clear eigenmode cache (for episodic reset)."""
        self._cached_modes = None
        self._cache_counter = 0
