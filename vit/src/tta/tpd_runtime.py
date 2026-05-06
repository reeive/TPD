from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .akc_controller import AKCDecision, AdaptiveKoopmanController
from .hur_controller import HURSplit, HankelUpdateRouter, RoutedUpdate
from .prompt_adapter import PromptParameterAdapter, PromptSnapshot
from .update_subspace import ProjectionState, UpdateSubspaceManager


@dataclass
class RuntimeCheckpoint:
    prompt: PromptSnapshot
    subspace_state: Dict
    akc_state: Dict
    step_index: int


@dataclass
class TPDStepResult:
    step: int
    basis_ready: bool
    mode: str
    rho: float
    scale: float
    rollback: bool
    stable: bool
    u_raw_norm: float
    u_applied_norm: float
    q_mean: float
    p_mean: float
    split: Optional[HURSplit]
    route: Optional[RoutedUpdate]
    decision: Optional[AKCDecision]
    projection_state: ProjectionState


class TPDRuntime:
    """Paper-aligned TPD runtime over prompt updates."""

    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 1e-3,
        subspace_rank: int = 16,
        window_size: int = 10,
        min_history: int = 2,
        akc_mode: str = "auto",
        rho_threshold: float = 1.0,
        rollback_patience: int = 3,
        min_scale: float = 0.1,
        full_condition_threshold: float = 1e5,
        full_improvement_margin: float = 0.05,
        min_mode_dwell: int = 2,
        hur_beta_c: float = 0.25,
        hur_beta_n: float = 0.1,
        hur_kappa: float = 2.0,
        hur_eps: float = 1e-6,
        hur_routing_mode: str = "fixed",
        hur_bn_max: float = 0.5,
        hur_bn_min: float = 0.05,
        hur_bc_tau: float = 5.0,
        projection_mode: str = "svd",
        basis_window: Optional[int] = None,
        anchor_lambda: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.base_lr = float(base_lr)
        self.anchor_lambda = float(anchor_lambda)

        self.prompt_adapter = PromptParameterAdapter(model)
        self.subspace = UpdateSubspaceManager(
            rank=subspace_rank,
            window_size=window_size,
            min_history=min_history,
            projection_mode=projection_mode,
            basis_window=basis_window,
            device=self.device,
        )
        self.akc = AdaptiveKoopmanController(
            mode=akc_mode,
            rho_threshold=rho_threshold,
            rollback_patience=rollback_patience,
            min_scale=min_scale,
            full_condition_threshold=full_condition_threshold,
            full_improvement_margin=full_improvement_margin,
            min_mode_dwell=min_mode_dwell,
            device=self.device,
        )
        self.hur = HankelUpdateRouter(
            beta_c=hur_beta_c,
            beta_n=hur_beta_n,
            kappa=hur_kappa,
            eps=hur_eps,
            routing_mode=hur_routing_mode,
            bn_max=hur_bn_max,
            bn_min=hur_bn_min,
            bc_tau=hur_bc_tau,
        )

        self.step_index = 0
        self._checkpoint: Optional[RuntimeCheckpoint] = None
        self._initial_prompt = self.prompt_adapter.snapshot()
        self._initial_akc_state = self.akc.state_dict()
        self._make_checkpoint()
        self._initial_prompt = self.prompt_adapter.snapshot()

    def zero_prompt_grad(self) -> None:
        for param in self.prompt_adapter.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def prompt_vector(self) -> torch.Tensor:
        return self.prompt_adapter.vector()

    def raw_update_from_grad(self, lr: Optional[float] = None) -> torch.Tensor:
        eta = self.base_lr if lr is None else float(lr)
        grad_vector = self.prompt_adapter.grad_vector()
        return -eta * grad_vector

    def _make_checkpoint(self) -> None:
        self._checkpoint = RuntimeCheckpoint(
            prompt=self.prompt_adapter.snapshot(),
            subspace_state=self.subspace.state_dict(),
            akc_state=self.akc.state_dict(),
            step_index=self.step_index,
        )

    def _restore_checkpoint(self) -> bool:
        if self._checkpoint is None:
            return False

        self.prompt_adapter.restore(self._checkpoint.prompt)
        self.subspace.load_state_dict(self._checkpoint.subspace_state)
        self.akc.load_state_dict(self._checkpoint.akc_state)
        self.step_index = self._checkpoint.step_index
        return True

    def reset_state(self, prompt_snapshot: Optional[PromptSnapshot] = None) -> None:
        target_snapshot = prompt_snapshot if prompt_snapshot is not None else self._initial_prompt
        self.prompt_adapter.restore(target_snapshot)
        self.subspace.clear()
        self.akc.load_state_dict(self._initial_akc_state)
        self.step_index = 0
        self._checkpoint = None
        self.zero_prompt_grad()
        self._make_checkpoint()

    def reset(self) -> None:
        self.prompt_adapter.restore(self._initial_prompt)
        self.subspace.clear()
        self.akc.reset()
        self._checkpoint = None
        self.step_index = 0

    def prompt_drift(self) -> float:
        current = self.prompt_adapter.vector()
        return float((current - self._initial_prompt.vector.to(current.device)).norm().item())

    def step(
        self,
        raw_update: Optional[torch.Tensor] = None,
        lr: Optional[float] = None,
    ) -> TPDStepResult:
        if raw_update is None:
            raw_update = self.raw_update_from_grad(lr=lr)
        raw_update = raw_update.detach().to(self.device)

        projection_state = self.subspace.fit_basis()
        basis_ready = projection_state.ready
        mode = "warmup"
        rho = 0.0
        scale = 1.0
        rollback = False
        stable = True
        q_mean = 0.0
        p_mean = 0.0
        split = None
        route = None
        decision = None

        applied_update = raw_update.clone()

        if basis_ready:
            basis = projection_state.basis
            projected_history = self.subspace.project_history(basis)
            decision = self.akc.estimate(projected_history)
            mode = decision.mode
            rho = decision.rho
            scale = decision.scale
            rollback = decision.rollback
            stable = decision.stable

            if (
                decision.diag_fit is not None
                and projected_history is not None
                and projected_history.shape[0] >= 1
                and basis is not None
            ):
                y_prev = projected_history[-1]
                y_curr, residual = self.subspace.project(raw_update, basis)
                split = self.hur.split(y_curr, y_prev, decision.diag_fit.coefficients)
                route = self.hur.route(
                    split, basis, residual,
                    diagonal_coefficients=decision.diag_fit.coefficients,
                )
                q_mean = float(split.explainability.mean().item())
                p_mean = float(split.persistence.mean().item())

                if rollback and self._restore_checkpoint():
                    return TPDStepResult(
                        step=self.step_index,
                        basis_ready=basis_ready,
                        mode=mode,
                        rho=rho,
                        scale=scale,
                        rollback=True,
                        stable=False,
                        u_raw_norm=float(raw_update.norm().item()),
                        u_applied_norm=0.0,
                        q_mean=q_mean,
                        p_mean=p_mean,
                        split=split,
                        route=route,
                        decision=decision,
                        projection_state=projection_state,
                    )

                applied_update = scale * route.routed

        self.subspace.append(raw_update)
        if self.anchor_lambda > 0:
            p_init = self._initial_prompt.vector.to(self.device)
            p_curr = self.prompt_adapter.vector()
            anchor_pull = self.anchor_lambda * (p_init - p_curr)
            self.prompt_adapter.apply_update(applied_update + anchor_pull)
        else:
            self.prompt_adapter.apply_update(applied_update)
        self.step_index += 1

        if stable and not rollback:
            self._make_checkpoint()

        return TPDStepResult(
            step=self.step_index,
            basis_ready=basis_ready,
            mode=mode,
            rho=rho,
            scale=scale,
            rollback=False,
            stable=stable,
            u_raw_norm=float(raw_update.norm().item()),
            u_applied_norm=float(applied_update.norm().item()),
            q_mean=q_mean,
            p_mean=p_mean,
            split=split,
            route=route,
            decision=decision,
            projection_state=projection_state,
        )

    def step_from_loss(
        self,
        loss: torch.Tensor,
        lr: Optional[float] = None,
        retain_graph: bool = False,
    ) -> TPDStepResult:
        self.model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=retain_graph)
        return self.step(lr=lr)
