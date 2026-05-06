from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


def _safe_linalg(fn, *args):
    """Run a torch.linalg function, falling back to CPU on CUSOLVER errors."""
    try:
        return fn(*args)
    except RuntimeError:
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        result = fn(*cpu_args)
        device = next((a.device for a in args if isinstance(a, torch.Tensor)), None)
        if device is not None and isinstance(result, torch.Tensor):
            return result.to(device)
        return result


@dataclass
class DiagonalKoopmanFit:
    coefficients: torch.Tensor
    stabilized: torch.Tensor
    rho: float
    prediction_error: float


@dataclass
class FullKoopmanFit:
    operator: torch.Tensor
    rho: float
    prediction_error: float
    condition_number: float
    rank: int


@dataclass
class AKCDecision:
    mode: str
    rho: float
    scale: float
    rollback: bool
    stable: bool
    diag_fit: Optional[DiagonalKoopmanFit]
    full_fit: Optional[FullKoopmanFit]
    full_reliable: bool


class AdaptiveKoopmanController:
    """Causal AKC controller with adaptive diagonal/full mode selection."""

    def __init__(
        self,
        mode: str = "auto",
        rho_threshold: float = 1.0,
        rollback_patience: int = 3,
        min_scale: float = 0.1,
        full_condition_threshold: float = 1e5,
        full_improvement_margin: float = 0.05,
        min_full_history: Optional[int] = None,
        min_mode_dwell: int = 2,
        eps: float = 1e-8,
        device: Optional[torch.device] = None,
    ):
        self.mode = mode
        self.rho_threshold = float(rho_threshold)
        self.rollback_patience = int(rollback_patience)
        self.min_scale = float(min_scale)
        self.full_condition_threshold = float(full_condition_threshold)
        self.full_improvement_margin = float(full_improvement_margin)
        self.min_full_history = min_full_history
        self.min_mode_dwell = int(min_mode_dwell)
        self.eps = float(eps)
        self.device = device or torch.device("cpu")

        self._current_mode = "diagonal" if mode == "auto" else mode
        self._mode_dwell = 0
        self._consecutive_unstable = 0

    def fit_diagonal(self, projected_history: torch.Tensor) -> DiagonalKoopmanFit:
        y_prev = projected_history[:-1].T.to(self.device)
        y_next = projected_history[1:].T.to(self.device)

        numerator = (y_prev * y_next).sum(dim=1)
        denominator = y_prev.square().sum(dim=1).clamp_min(self.eps)
        coefficients = numerator / denominator
        prediction = coefficients.unsqueeze(1) * y_prev
        error = torch.mean((y_next - prediction).square()).item()
        rho = float(coefficients.abs().max().item()) if coefficients.numel() > 0 else 0.0
        clamp_ratio = torch.clamp(
            self.rho_threshold / coefficients.abs().clamp_min(self.eps), max=1.0)
        stabilized = coefficients * clamp_ratio
        return DiagonalKoopmanFit(
            coefficients=coefficients, stabilized=stabilized,
            rho=rho, prediction_error=error)

    def fit_full(self, projected_history: torch.Tensor) -> Optional[FullKoopmanFit]:
        y_prev = projected_history[:-1].T.to(self.device)
        y_next = projected_history[1:].T.to(self.device)
        if y_prev.numel() == 0:
            return None

        pinv_prev = _safe_linalg(torch.linalg.pinv, y_prev)
        operator = y_next @ pinv_prev
        prediction = operator @ y_prev
        error = torch.mean((y_next - prediction).square()).item()

        gram = y_prev @ y_prev.T
        singular_values = _safe_linalg(torch.linalg.svdvals, gram)
        if singular_values.numel() == 0:
            condition_number = float("inf")
        else:
            s_max = float(singular_values.max().item())
            s_min = float(singular_values.min().item())
            condition_number = float("inf") if s_min <= self.eps else s_max / s_min

        eigenvalues = _safe_linalg(torch.linalg.eigvals, operator)
        rho = float(eigenvalues.abs().max().item()) if eigenvalues.numel() > 0 else 0.0
        rank = int(_safe_linalg(torch.linalg.matrix_rank, y_prev).item())

        return FullKoopmanFit(
            operator=operator,
            rho=rho,
            prediction_error=error,
            condition_number=condition_number,
            rank=rank,
        )

    def _select_mode(
        self,
        projected_history: torch.Tensor,
        diag_fit: DiagonalKoopmanFit,
        full_fit: Optional[FullKoopmanFit],
    ):
        if self.mode in {"diagonal", "full"}:
            selected = self.mode
            full_reliable = full_fit is not None
            self._current_mode = selected
            self._mode_dwell += 1
            return selected, full_reliable

        history_len, projected_rank = projected_history.shape
        min_full_history = (
            self.min_full_history
            if self.min_full_history is not None
            else projected_rank + 1
        )

        full_reliable = False
        if full_fit is not None:
            target_rank = min(projected_rank, max(history_len - 1, 1))
            full_reliable = (
                history_len >= min_full_history
                and full_fit.rank >= target_rank
                and torch.isfinite(torch.tensor(full_fit.condition_number))
                and full_fit.condition_number <= self.full_condition_threshold
            )

        desired_mode = "diagonal"
        if full_reliable and full_fit is not None:
            materially_better = (
                full_fit.prediction_error
                <= diag_fit.prediction_error * (1.0 - self.full_improvement_margin)
            )
            if self._current_mode == "full":
                desired_mode = "full"
            elif materially_better:
                desired_mode = "full"

        if desired_mode != self._current_mode and self._mode_dwell < self.min_mode_dwell:
            desired_mode = self._current_mode
        elif desired_mode != self._current_mode:
            self._current_mode = desired_mode
            self._mode_dwell = 0

        self._mode_dwell += 1
        return desired_mode, full_reliable

    def estimate(self, projected_history: Optional[torch.Tensor]) -> AKCDecision:
        if projected_history is None or projected_history.shape[0] < 2:
            return AKCDecision(
                mode=self._current_mode,
                rho=0.0,
                scale=1.0,
                rollback=False,
                stable=True,
                diag_fit=None,
                full_fit=None,
                full_reliable=False,
            )

        diag_fit = self.fit_diagonal(projected_history)
        full_fit = None if self.mode == "diagonal" else self.fit_full(projected_history)
        selected_mode, full_reliable = self._select_mode(projected_history, diag_fit, full_fit)

        if selected_mode == "full" and full_fit is not None:
            rho = full_fit.rho
        else:
            rho = diag_fit.rho

        stable = rho <= self.rho_threshold
        if stable:
            self._consecutive_unstable = 0
        else:
            self._consecutive_unstable += 1

        rollback = self._consecutive_unstable >= self.rollback_patience and not stable
        if stable:
            scale = 1.0
        else:
            scale = max(self.min_scale, self.rho_threshold / max(rho, self.eps))

        return AKCDecision(
            mode=selected_mode,
            rho=rho,
            scale=scale,
            rollback=rollback,
            stable=stable,
            diag_fit=diag_fit,
            full_fit=full_fit,
            full_reliable=full_reliable,
        )

    def state_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "current_mode": self._current_mode,
            "mode_dwell": self._mode_dwell,
            "consecutive_unstable": self._consecutive_unstable,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.mode = state_dict.get("mode", self.mode)
        self._current_mode = state_dict.get("current_mode", self._current_mode)
        self._mode_dwell = int(state_dict.get("mode_dwell", 0))
        self._consecutive_unstable = int(state_dict.get("consecutive_unstable", 0))

    def reset(self) -> None:
        self._current_mode = "diagonal" if self.mode == "auto" else self.mode
        self._mode_dwell = 0
        self._consecutive_unstable = 0
