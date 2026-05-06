from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import torch


@dataclass
class ProjectionState:
    basis: Optional[torch.Tensor]
    singular_values: Optional[torch.Tensor]
    energy: Optional[torch.Tensor]
    history_length: int
    rank: int
    ready: bool


class UpdateSubspaceManager:
    """Build the projected update subspace from recent updates.

    projection_mode:
        "svd"    — Q_t from SVD of update history.
                   basis_window (B) controls how many past updates feed the SVD.
                   Effective rank = min(r, B).  When B >= r, rank is NOT
                   truncated by the Koopman window W.
        "random" — fixed random orthogonal basis Q ∈ R^{d×r}.

    Two separate deques are maintained:
        _basis_history  (maxlen = basis_window)  — feeds SVD basis construction
        _koopman_history (maxlen = window_size)  — feeds Koopman DMD via
                                                   project_history()
    """

    def __init__(
        self,
        rank: int,
        window_size: int,
        min_history: int = 2,
        projection_mode: str = "svd",
        basis_window: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.rank = int(rank)
        self.window_size = int(window_size)
        self.min_history = int(min_history)
        self.projection_mode = projection_mode
        self.device = device or torch.device("cpu")

        # basis_window defaults to window_size (legacy behaviour)
        self.basis_window = int(basis_window) if basis_window is not None else self.window_size

        self._basis_history: Deque[torch.Tensor] = deque(maxlen=self.basis_window)
        self._koopman_history: Deque[torch.Tensor] = deque(maxlen=self.window_size)

        self._random_basis: Optional[torch.Tensor] = None
        self._state = ProjectionState(
            basis=None,
            singular_values=None,
            energy=None,
            history_length=0,
            rank=self.rank,
            ready=False,
        )

    def append(self, update: torch.Tensor) -> None:
        vec = update.detach().to(self.device).clone()
        self._basis_history.append(vec)
        self._koopman_history.append(vec)
        if self.projection_mode == "random" and self._random_basis is None:
            d = update.shape[0]
            r = min(self.rank, d)
            raw = torch.randn(d, r, dtype=update.dtype)
            q, _ = torch.linalg.qr(raw)
            self._random_basis = q[:, :r].to(self.device)

    def clear(self) -> None:
        self._basis_history.clear()
        self._koopman_history.clear()
        self._state = ProjectionState(
            basis=None,
            singular_values=None,
            energy=None,
            history_length=0,
            rank=self.rank,
            ready=False,
        )
        if self.projection_mode != "random":
            self._random_basis = None

    def history_length(self) -> int:
        return len(self._koopman_history)

    def basis_history_length(self) -> int:
        return len(self._basis_history)

    def history_tensor(self) -> Optional[torch.Tensor]:
        """Koopman window (last W updates)."""
        if not self._koopman_history:
            return None
        return torch.stack(list(self._koopman_history), dim=0).to(self.device)

    def basis_history_tensor(self) -> Optional[torch.Tensor]:
        """Full basis window (last B updates) for SVD."""
        if not self._basis_history:
            return None
        return torch.stack(list(self._basis_history), dim=0).to(self.device)

    def fit_basis(self) -> ProjectionState:
        basis_hist = self.basis_history_tensor()
        koopman_hist = self.history_tensor()
        min_len = 0 if basis_hist is None else int(basis_hist.shape[0])

        if basis_hist is None or min_len < self.min_history:
            self._state = ProjectionState(
                basis=None,
                singular_values=None,
                energy=None,
                history_length=min_len,
                rank=self.rank,
                ready=False,
            )
            return self._state

        if self.projection_mode == "random" and self._random_basis is not None:
            basis = self._random_basis
            k = basis.shape[1]
            self._state = ProjectionState(
                basis=basis,
                singular_values=None,
                energy=None,
                history_length=min_len,
                rank=int(k),
                ready=True,
            )
            return self._state

        # SVD on the (larger) basis history — rank up to min(r, B)
        history_matrix = basis_hist.T          # d × B
        try:
            u, s, _ = torch.linalg.svd(history_matrix, full_matrices=False)
        except RuntimeError:
            hm_cpu = history_matrix.cpu()
            u_cpu, s_cpu, _ = torch.linalg.svd(hm_cpu, full_matrices=False)
            u = u_cpu.to(history_matrix.device)
            s = s_cpu.to(history_matrix.device)
        k = min(self.rank, u.shape[1])
        basis = u[:, :k]

        energy = s.square()
        if energy.numel() > 0:
            energy = energy / energy.sum().clamp_min(1e-12)

        self._state = ProjectionState(
            basis=basis,
            singular_values=s[:k],
            energy=energy[:k] if energy.numel() > 0 else None,
            history_length=min_len,
            rank=int(k),
            ready=True,
        )
        return self._state

    def project(self, update: torch.Tensor, basis: Optional[torch.Tensor] = None):
        basis = basis if basis is not None else self._state.basis
        update = update.detach().to(self.device)
        if basis is None:
            return None, update.clone()
        y = basis.T @ update
        residual = update - basis @ y
        return y, residual

    def project_history(self, basis: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Project the *Koopman* window (last W updates) onto the basis."""
        history = self.history_tensor()   # W × d
        basis = basis if basis is not None else self._state.basis
        if history is None or basis is None:
            return None
        return history @ basis            # W × r

    def reconstruct(
        self,
        y: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        basis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        basis = basis if basis is not None else self._state.basis
        if basis is None:
            if residual is None:
                raise ValueError("Cannot reconstruct without either a basis or residual.")
            return residual.detach().clone()
        recon = basis @ y
        if residual is not None:
            recon = recon + residual
        return recon

    def state_dict(self) -> Dict:
        sd = {
            "rank": self.rank,
            "window_size": self.window_size,
            "basis_window": self.basis_window,
            "min_history": self.min_history,
            "projection_mode": self.projection_mode,
            "basis_history": [item.detach().clone() for item in self._basis_history],
            "koopman_history": [item.detach().clone() for item in self._koopman_history],
        }
        if self._random_basis is not None:
            sd["random_basis"] = self._random_basis.detach().clone()
        return sd

    def load_state_dict(self, state_dict: Dict) -> None:
        self.rank = int(state_dict["rank"])
        self.window_size = int(state_dict["window_size"])
        self.basis_window = int(state_dict.get("basis_window", self.window_size))
        self.min_history = int(state_dict["min_history"])
        self.projection_mode = state_dict.get("projection_mode", self.projection_mode)
        # backward compat: old checkpoints have "history" instead of split
        if "koopman_history" in state_dict:
            self._koopman_history = deque(
                [v.detach().to(self.device).clone() for v in state_dict["koopman_history"]],
                maxlen=self.window_size,
            )
            self._basis_history = deque(
                [v.detach().to(self.device).clone() for v in state_dict["basis_history"]],
                maxlen=self.basis_window,
            )
        else:
            vecs = [v.detach().to(self.device).clone() for v in state_dict.get("history", [])]
            self._koopman_history = deque(vecs, maxlen=self.window_size)
            self._basis_history = deque(vecs, maxlen=self.basis_window)
        if "random_basis" in state_dict:
            self._random_basis = state_dict["random_basis"].to(self.device)
        self.fit_basis()
