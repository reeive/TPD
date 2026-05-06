"""GPU-friendly projected subspace: QR + small SVD on R (B×B) instead of tall-thin SVD on CPU."""
from __future__ import annotations

import torch

from .subspace import ProjectedUpdateSubspace, SubspaceState


class ProjectedUpdateSubspaceGPU(ProjectedUpdateSubspace):
    """Same API as ProjectedUpdateSubspace; fit_basis uses QR(H) + SVD(R) on self.device."""

    def fit_basis(self) -> SubspaceState:
        if len(self._basis_buf) < self.min_history:
            return SubspaceState(
                basis=None, singular_values=None, rank=self.r, ready=False
            )
        H = torch.stack(list(self._basis_buf), dim=0).T  # d × B
        H = H.to(device=self.device, dtype=torch.float32)
        try:
            Q_econ, R = torch.linalg.qr(H, mode="reduced")
            U_r, S, _ = torch.linalg.svd(R, full_matrices=False)
        except RuntimeError:
            U, S, _ = torch.linalg.svd(H.cpu(), full_matrices=False)
            U, S = U.to(self.device), S.to(self.device)
            k = min(self.r, U.shape[1])
            return SubspaceState(
                basis=U[:, :k],
                singular_values=S[:k],
                rank=int(k),
                ready=True,
            )
        k = min(self.r, U_r.shape[1])
        basis = Q_econ @ U_r[:, :k]
        return SubspaceState(
            basis=basis,
            singular_values=S[:k],
            rank=int(k),
            ready=True,
        )
