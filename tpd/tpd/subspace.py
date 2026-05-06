"""§2.2 Projected update subspace (SVD basis + Koopman window)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import torch


@dataclass
class SubspaceState:
    basis: Optional[torch.Tensor]  # Q_t ∈ R^{d×k}
    singular_values: Optional[torch.Tensor]
    rank: int  # effective k = min(r, B)
    ready: bool


class ProjectedUpdateSubspace:
    """SVD subspace on history length B; Koopman uses history length W."""

    def __init__(
        self,
        r: int,
        W: int,
        B: Optional[int] = None,
        min_history: int = 2,
        device: Optional[torch.device] = None,
    ):
        self.r = int(r)
        self.W = int(W)
        self.B = int(B) if B is not None else self.W
        self.min_history = int(min_history)
        self.device = device or torch.device("cpu")
        self._basis_buf: Deque[torch.Tensor] = deque(maxlen=self.B)
        self._koopman_buf: Deque[torch.Tensor] = deque(maxlen=self.W)

    def append(self, u: torch.Tensor) -> None:
        v = u.detach().to(self.device).clone()
        self._basis_buf.append(v)
        self._koopman_buf.append(v)

    def clear(self) -> None:
        self._basis_buf.clear()
        self._koopman_buf.clear()

    def fit_basis(self) -> SubspaceState:
        if len(self._basis_buf) < self.min_history:
            return SubspaceState(
                basis=None, singular_values=None, rank=self.r, ready=False
            )
        H = torch.stack(list(self._basis_buf), dim=0).T  # d × B
        # Large update dimension: GPU SVD workspace can OOM; use CPU (same as vit fallback).
        if H.shape[0] > 4096:
            U, S, _ = torch.linalg.svd(H.cpu(), full_matrices=False)
            U, S = U.to(self.device), S.to(self.device)
        else:
            try:
                U, S, _ = torch.linalg.svd(H, full_matrices=False)
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

    def project(self, u: torch.Tensor, Q: torch.Tensor):
        u = u.detach().to(self.device)
        y = Q.T @ u
        e = u - Q @ y
        return y, e

    def project_history(self, Q: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._koopman_buf:
            return None
        H_k = torch.stack(list(self._koopman_buf), dim=0).to(self.device)
        return H_k @ Q  # W × r

    def state_dict(self) -> Dict:
        return {
            "r": self.r,
            "W": self.W,
            "B": self.B,
            "min_history": self.min_history,
            "basis_buf": [v.clone() for v in self._basis_buf],
            "koopman_buf": [v.clone() for v in self._koopman_buf],
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.r, self.W, self.B = int(sd["r"]), int(sd["W"]), int(sd["B"])
        self.min_history = int(sd["min_history"])
        self._basis_buf = deque(
            (v.to(self.device) for v in sd["basis_buf"]),
            maxlen=self.B,
        )
        self._koopman_buf = deque(
            (v.to(self.device) for v in sd["koopman_buf"]),
            maxlen=self.W,
        )
