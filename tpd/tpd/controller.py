"""TPD main orchestrator: subspace + AKC + HUR + optional anchor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .akc import AdaptiveKoopmanControl
from .hur import HURDecomposition, HankelUpdateRouter
from .prompt import PromptParameterAdapter
from .subspace import ProjectedUpdateSubspace, SubspaceState


@dataclass
class TPDStepInfo:
    step: int
    rho: float
    scale: float
    rollback: bool
    q_mean: float
    p_mean: float
    u_raw_norm: float
    u_applied_norm: float
    f1: float = 0.0
    f2: float = 0.0
    f3: float = 0.0
    svd_spectrum: List[float] = field(
        default_factory=lambda: [0.0] * 20
    )
    # Extended decomposition / logging (see tta_decomposition_logging_requirements.md)
    is_warmup: bool = True
    r_eff: int = 0
    singular_values: List[float] = field(default_factory=list)
    projected_norm: float = 0.0
    off_subspace_residual_norm: float = 0.0
    projection_residual_ratio: float = 0.0
    S_norm: float = 0.0
    C_norm: float = 0.0
    N_norm: float = 0.0
    persistent_ratio: float = 0.0
    oscillatory_ratio: float = 0.0
    residual_ratio: float = 0.0
    hur_update_norm: float = 0.0
    controlled_update_norm: float = 0.0
    magnitude_reduction_ratio: float = 1.0
    cos_raw_hur: float = 1.0
    cos_raw_controlled: float = 1.0


def _cosine_tensors(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> float:
    nu = float(u.detach().norm().item())
    nv = float(v.detach().norm().item())
    if nu < eps or nv < eps:
        return 0.0
    return float((u.detach().flatten() @ v.detach().flatten()).item() / (nu * nv))


def _cosine_tensors_batched(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> float:
    """One CPU sync for cosine(u, v) (cheaper than separate norm + dot .item() calls)."""
    with torch.no_grad():
        uf = u.detach().flatten().float()
        vf = v.detach().flatten().float()
        pack = torch.stack([(uf * uf).sum(), (vf * vf).sum(), (uf * vf).sum()]).cpu()
    nu2, nv2, dot = float(pack[0]), float(pack[1]), float(pack[2])
    nu = nu2**0.5
    nv = nv2**0.5
    if nu < eps or nv < eps:
        return 0.0
    return dot / (nu * nv)


class TPD:
    """Paper-aligned TPD controller for prompt-based TTA."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.025,
        r: int = 16,
        W: int = 10,
        B: Optional[int] = None,
        K: int = 3,
        rho_threshold: float = 1.0,
        rollback_patience: int = 3,
        beta_c: float = 0.25,
        beta_n: float = 0.1,
        kappa: float = 2.0,
        anchor_lambda: float = 0.0,
        device: Optional[torch.device] = None,
        opt_mode: str = "sgd",
        ablation_mode: str = "tpd_full",
        defer_logging_sync: bool = True,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.lr = float(lr)
        self.K = int(K)
        self.anchor_lambda = float(anchor_lambda)
        self.opt_mode = str(opt_mode or "sgd")
        self.ablation_mode = (ablation_mode or "tpd_full").lower()
        self.defer_logging_sync = bool(defer_logging_sync)
        if self.ablation_mode not in (
            "tpd_full",
            "akc_diag",
            "akc_full",
            "hur_plain",
            "hur_hankel",
        ):
            raise ValueError(
                f"Unknown TPD ablation_mode: {ablation_mode!r}"
            )
        # AKC: diagonal DMD (default / tpd / hur coeff) or full r×r Koopman
        _akc_mode = (
            "full_koopman" if self.ablation_mode == "akc_full" else "diagonal"
        )
        # HUR: Hankel routing (default) or plain subspace back-projection
        _hur_mode = (
            "plain" if self.ablation_mode == "hur_plain" else "hankel"
        )

        self.prompt = PromptParameterAdapter(model)
        self.subspace = ProjectedUpdateSubspace(
            r=r, W=W, B=B, device=self.device
        )
        self.akc = AdaptiveKoopmanControl(
            rho_threshold=rho_threshold,
            patience=rollback_patience,
            mode=_akc_mode,
        )
        self.hur = HankelUpdateRouter(
            beta_c=beta_c, beta_n=beta_n, kappa=kappa, mode=_hur_mode
        )

        self.step_idx = 0
        self._ckpt: Optional[Dict] = None
        self._init_snap = self.prompt.snapshot()
        self.collect_vectors: bool = False
        self.last_vectors: Optional[Dict[str, torch.Tensor]] = None
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        self._ckpt = {
            "prompt": self.prompt.snapshot(),
            "subspace": self.subspace.state_dict(),
            "akc": self.akc.state_dict(),
            "step": self.step_idx,
        }

    def _rollback(self) -> None:
        if self._ckpt is None:
            return
        self.prompt.restore(self._ckpt["prompt"])
        self.subspace.load_state_dict(self._ckpt["subspace"])
        self.akc.load_state_dict(self._ckpt["akc"])
        self.step_idx = self._ckpt["step"]

    def reset(self) -> None:
        self.prompt.restore(self._init_snap)
        self.subspace.clear()
        self.akc.reset()
        self.step_idx = 0
        self._ckpt = None
        self._save_checkpoint()

    def _normalized_spectrum_20(
        self, singular_values: Optional[torch.Tensor]
    ) -> List[float]:
        if singular_values is None:
            return [0.0] * 20
        s = singular_values.detach().float().cpu().view(-1)[:20]
        pad = torch.zeros(20, dtype=s.dtype)
        pad[: s.numel()] = s
        pad = torch.clamp(pad, min=0.0)
        tot = float(pad.sum().item()) + 1e-12
        return [float(x) / tot for x in pad.tolist()]

    def _raw_from_grad(self, g: torch.Tensor) -> torch.Tensor:
        """Map gradient to raw update; ``sgd`` matches the original TPD paper."""
        g = g.detach().to(self.device)
        mode = (self.opt_mode or "sgd").lower()
        lr = self.lr
        if mode in ("sgd", "default"):
            out = -lr * g
        elif mode == "signsgd":
            out = -lr * torch.sign(g)
        elif mode == "adam_init":
            out = -lr * g / (g.abs() + 1e-8)
        else:
            raise ValueError(
                f"Unknown TPD opt_mode: {self.opt_mode!r} (use sgd, signsgd, adam_init)"
            )
        return out.to(dtype=g.dtype)

    @staticmethod
    def _energy_ratios(
        f1: float, f2: float, f3: float, eps: float = 1e-12
    ) -> Tuple[float, float, float]:
        tot = f1 + f2 + f3
        if tot < eps:
            return 0.0, 0.0, 0.0
        tot = tot + eps
        return f1 / tot, f2 / tot, f3 / tot

    def _snapshot_vectors(
        self,
        raw: torch.Tensor,
        Q: torch.Tensor,
        y_curr: torch.Tensor,
        y_prev: Optional[torch.Tensor],
        e: torch.Tensor,
        a: Optional[torch.Tensor],
        decomp: Optional[HURDecomposition],
        S_vec: Optional[torch.Tensor],
        C_vec: Optional[torch.Tensor],
        N_vec: Optional[torch.Tensor],
        u_hur: Optional[torch.Tensor],
        delta_p: Optional[torch.Tensor],
    ) -> None:
        if not getattr(self, "collect_vectors", False):
            self.last_vectors = None
            return
        out: Dict[str, torch.Tensor] = {
            "u": raw.detach().float().cpu().clone(),
            "projected_update": (Q @ y_curr).detach().float().cpu().clone(),
            "e": e.detach().float().cpu().clone(),
            "y": y_curr.detach().float().cpu().clone(),
        }
        if y_prev is not None and a is not None:
            y_hat = a * y_prev
            out["y_hat"] = y_hat.detach().float().cpu().clone()
            out["r"] = (y_curr - y_hat).detach().float().cpu().clone()
            out["a"] = a.detach().float().cpu().clone()
        if decomp is not None:
            out["s_star"] = decomp.persistent.detach().float().cpu().clone()
            out["c_star"] = decomp.oscillatory.detach().float().cpu().clone()
            out["n_star"] = decomp.noise.detach().float().cpu().clone()
            out["q"] = decomp.q.detach().float().cpu().clone()
            out["pi"] = decomp.p.detach().float().cpu().clone()
        if S_vec is not None:
            out["S"] = S_vec.detach().float().cpu().clone()
        if C_vec is not None:
            out["C"] = C_vec.detach().float().cpu().clone()
        if N_vec is not None:
            out["N"] = N_vec.detach().float().cpu().clone()
        if u_hur is not None:
            out["u_hur"] = u_hur.detach().float().cpu().clone()
        if delta_p is not None:
            out["delta_p"] = delta_p.detach().float().cpu().clone()
        self.last_vectors = out

    def _make_step_info(
        self,
        raw: torch.Tensor,
        applied_final: torch.Tensor,
        spec20: List[float],
        sv_list: List[float],
        *,
        rho: float,
        scale: float,
        rollback: bool,
        q_mean: float,
        p_mean: float,
        f1: float,
        f2: float,
        f3: float,
        state: SubspaceState,
        is_warmup: bool,
        projected_norm: float,
        off_norm: float,
        u_raw_n: float,
        hur_norm: float,
        controlled_norm: float,
        Q: Optional[torch.Tensor],
        y_curr: Optional[torch.Tensor],
        y_prev: Optional[torch.Tensor],
        e: Optional[torch.Tensor],
        a_coeff: Optional[torch.Tensor],
        decomp: Optional[HURDecomposition],
        S_vec: Optional[torch.Tensor],
        C_vec: Optional[torch.Tensor],
        N_vec: Optional[torch.Tensor],
        u_hur: Optional[torch.Tensor],
        delta_p: Optional[torch.Tensor],
        eps: float = 1e-8,
    ) -> TPDStepInfo:
        pr, otr, rr = self._energy_ratios(f1, f2, f3)
        proj_ratio = off_norm / (u_raw_n + eps) if u_raw_n > 0 else 0.0
        if (
            self.defer_logging_sync
            and S_vec is not None
            and C_vec is not None
            and N_vec is not None
        ):
            t_sn = torch.stack(
                [
                    S_vec.norm(),
                    C_vec.norm(),
                    N_vec.norm(),
                    applied_final.norm(),
                ]
            ).detach().float().cpu()
            s_n, c_n, n_n, u_applied_n = (float(x) for x in t_sn)
        else:
            s_n = float((S_vec.norm().item()) if S_vec is not None else 0.0)
            c_n = float((C_vec.norm().item()) if C_vec is not None else 0.0)
            n_n = float((N_vec.norm().item()) if N_vec is not None else 0.0)
            u_applied_n = float(applied_final.norm().item())
        mag_red = controlled_norm / (u_raw_n + eps) if u_raw_n > 0 else 0.0
        if self.defer_logging_sync:
            c_h = (
                _cosine_tensors_batched(raw, u_hur, eps)
                if u_hur is not None
                else (1.0 if rollback else 0.0)
            )
            c_c = (
                _cosine_tensors_batched(raw, delta_p, eps)
                if delta_p is not None
                else (c_h if not rollback else 0.0)
            )
        else:
            c_h = (
                _cosine_tensors(raw, u_hur, eps)
                if u_hur is not None
                else (1.0 if rollback else 0.0)
            )
            c_c = (
                _cosine_tensors(raw, delta_p, eps)
                if delta_p is not None
                else (c_h if not rollback else 0.0)
            )
        if (
            not rollback
            and u_hur is None
            and (f1 + f2 + f3) < eps
            and controlled_norm < eps
        ):
            c_h = 1.0
            c_c = 1.0
            mag_red = 1.0
            hur_norm = u_raw_n
            controlled_norm = u_raw_n
        if Q is not None and y_curr is not None and e is not None:
            self._snapshot_vectors(
                raw,
                Q,
                y_curr,
                y_prev,
                e,
                a_coeff,
                decomp,
                S_vec,
                C_vec,
                N_vec,
                u_hur,
                delta_p,
            )
        else:
            self.last_vectors = None
        r_eff = int(state.rank) if state.rank is not None else 0
        return TPDStepInfo(
            step=self.step_idx,
            rho=float(rho),
            scale=float(scale),
            rollback=bool(rollback),
            q_mean=float(q_mean),
            p_mean=float(p_mean),
            u_raw_norm=float(u_raw_n),
            u_applied_norm=u_applied_n,
            f1=float(f1),
            f2=float(f2),
            f3=float(f3),
            svd_spectrum=list(spec20),
            is_warmup=bool(is_warmup),
            r_eff=r_eff,
            singular_values=list(sv_list),
            projected_norm=float(projected_norm),
            off_subspace_residual_norm=float(off_norm),
            projection_residual_ratio=float(proj_ratio),
            S_norm=s_n,
            C_norm=c_n,
            N_norm=n_n,
            persistent_ratio=float(pr),
            oscillatory_ratio=float(otr),
            residual_ratio=float(rr),
            hur_update_norm=float(hur_norm),
            controlled_update_norm=float(controlled_norm),
            magnitude_reduction_ratio=float(mag_red),
            cos_raw_hur=float(c_h),
            cos_raw_controlled=float(c_c),
        )

    def _rollback_info(
        self,
        raw: torch.Tensor,
        spec20: List[float],
        sv_list: List[float],
        state: SubspaceState,
        *,
        rho: float,
        scale: float,
        u_raw_n: float,
        projected_norm: float,
        off_norm: float,
        is_warmup: bool,
    ) -> TPDStepInfo:
        self.last_vectors = None
        proj_ratio = off_norm / (u_raw_n + 1e-8) if u_raw_n > 0 else 0.0
        return TPDStepInfo(
            step=self.step_idx,
            rho=float(rho),
            scale=float(scale),
            rollback=True,
            q_mean=0.0,
            p_mean=0.0,
            u_raw_norm=float(u_raw_n),
            u_applied_norm=0.0,
            f1=0.0,
            f2=0.0,
            f3=0.0,
            svd_spectrum=list(spec20),
            is_warmup=bool(is_warmup),
            r_eff=int(state.rank) if state.rank is not None else 0,
            singular_values=list(sv_list),
            projected_norm=float(projected_norm),
            off_subspace_residual_norm=float(off_norm),
            projection_residual_ratio=float(proj_ratio),
            S_norm=0.0,
            C_norm=0.0,
            N_norm=0.0,
            persistent_ratio=0.0,
            oscillatory_ratio=0.0,
            residual_ratio=0.0,
            hur_update_norm=0.0,
            controlled_update_norm=0.0,
            magnitude_reduction_ratio=0.0,
            cos_raw_hur=0.0,
            cos_raw_controlled=0.0,
        )

    def _step(self) -> TPDStepInfo:
        self.last_vectors = None
        g = self.prompt.grad_vector()
        raw = self._raw_from_grad(g)
        u_raw_n = float(raw.norm().item())
        eps = 1e-8

        state = self.subspace.fit_basis()
        spec20 = self._normalized_spectrum_20(state.singular_values)
        sv_list: List[float] = []
        if state.singular_values is not None:
            sv_list = [
                float(x)
                for x in state.singular_values.detach().cpu().view(-1).tolist()
            ]

        applied = raw.clone()
        rho, scale, rollback = 0.0, 1.0, False
        q_mean, p_mean = 0.0, 0.0
        f1 = f2 = f3 = 0.0
        stable = True
        am = self.ablation_mode

        projected_norm = 0.0
        off_norm = 0.0
        hur_norm = 0.0
        controlled_norm = 0.0

        Q = None
        y_curr = None
        y_prev = None
        e = None
        decomp = None
        a_coeff = None
        S_vec = None
        C_vec = None
        N_vec = None
        u_hur = None
        delta_p = None
        Y = None

        koop_ready = False
        if state.ready and state.basis is not None:
            Q = state.basis
            Y = self.subspace.project_history(Q)
            y_curr, e = self.subspace.project(raw, Q)
            y_prev = Y[-1] if (Y is not None and Y.shape[0] >= 1) else y_curr
            proj_u = Q @ y_curr
            if self.defer_logging_sync:
                t2 = torch.stack([proj_u.norm(), e.norm()]).detach().float().cpu()
                projected_norm = float(t2[0])
                off_norm = float(t2[1])
            else:
                projected_norm = float(proj_u.norm().item())
                off_norm = float(e.norm().item())
            koop_ready = Y is not None and Y.shape[0] >= 2

        is_warmup = not state.ready or not koop_ready

        if state.ready and state.basis is not None and Q is not None:
            assert y_curr is not None and e is not None

            if am in ("akc_diag", "akc_full"):
                akc_out = self.akc.estimate(Y)
                if akc_out is not None and Y is not None and Y.shape[0] >= 1:
                    rho, scale = akc_out.rho, akc_out.scale
                    rollback = akc_out.rollback
                    stable = akc_out.stable
                    if rollback:
                        self._rollback()
                        self.subspace.append(raw)
                        self.step_idx += 1
                        return self._rollback_info(
                            raw,
                            spec20,
                            sv_list,
                            state,
                            rho=rho,
                            scale=scale,
                            u_raw_n=u_raw_n,
                            projected_norm=projected_norm,
                            off_norm=off_norm,
                            is_warmup=is_warmup,
                        )
                    u_rec = Q @ y_curr + e
                    u_hur = u_rec
                    delta_p = scale * u_rec
                    applied = delta_p.clone()
                    if self.defer_logging_sync:
                        t2 = torch.stack([u_rec.norm(), delta_p.norm()]).detach().float().cpu()
                        hur_norm = float(t2[0])
                        controlled_norm = float(t2[1])
                    else:
                        hur_norm = float(u_rec.norm().item())
                        controlled_norm = float(delta_p.norm().item())

            elif am == "hur_plain":
                r_plain = self.hur.route_plain(Q, y_curr, e)
                u_hur = r_plain.routed
                delta_p = u_hur.clone()
                applied = u_hur.clone()
                scale = 1.0
                hur_norm = float(u_hur.norm().item())
                controlled_norm = hur_norm

            elif am == "hur_hankel":
                a = (
                    self.akc.diagonal_dmd_a(Y, self.akc.eps)
                    if Y is not None
                    else None
                )
                if a is not None:
                    rho = float(a.abs().max().item())
                    a_coeff = a
                    decomp = self.hur.decompose(y_curr, y_prev, a)
                    routed = self.hur.route(decomp, Q, e)
                    u_hur = routed.routed
                    delta_p = u_hur.clone()
                    applied = u_hur.clone()
                    scale = 1.0
                    q_mean, p_mean = routed.q_mean, routed.p_mean
                    S_vec = Q @ decomp.persistent
                    C_vec = Q @ decomp.oscillatory
                    N_vec = Q @ decomp.noise + e
                    if self.defer_logging_sync:
                        t4 = torch.stack(
                            [
                                S_vec.pow(2).sum(),
                                C_vec.pow(2).sum(),
                                N_vec.pow(2).sum(),
                                u_hur.norm(),
                            ]
                        ).detach().float().cpu()
                        f1, f2, f3, hur_norm = map(float, t4)
                        controlled_norm = hur_norm
                    else:
                        f1 = float(S_vec.pow(2).sum().item())
                        f2 = float(C_vec.pow(2).sum().item())
                        f3 = float(N_vec.pow(2).sum().item())
                        hur_norm = float(u_hur.norm().item())
                        controlled_norm = hur_norm

            else:  # tpd_full
                akc_out = self.akc.estimate(Y)
                if akc_out is not None and Y is not None and Y.shape[0] >= 1:
                    rho, scale = akc_out.rho, akc_out.scale
                    rollback = akc_out.rollback
                    stable = akc_out.stable
                    if rollback:
                        self._rollback()
                        self.subspace.append(raw)
                        self.step_idx += 1
                        return self._rollback_info(
                            raw,
                            spec20,
                            sv_list,
                            state,
                            rho=rho,
                            scale=scale,
                            u_raw_n=u_raw_n,
                            projected_norm=projected_norm,
                            off_norm=off_norm,
                            is_warmup=is_warmup,
                        )
                    a_coeff = akc_out.coefficients
                    decomp = self.hur.decompose(
                        y_curr, y_prev, akc_out.coefficients
                    )
                    routed = self.hur.route(decomp, Q, e)
                    u_hur = routed.routed
                    delta_p = scale * u_hur
                    applied = delta_p.clone()
                    q_mean, p_mean = routed.q_mean, routed.p_mean
                    S_vec = Q @ decomp.persistent
                    C_vec = Q @ decomp.oscillatory
                    N_vec = Q @ decomp.noise + e
                    if self.defer_logging_sync:
                        t5 = torch.stack(
                            [
                                S_vec.pow(2).sum(),
                                C_vec.pow(2).sum(),
                                N_vec.pow(2).sum(),
                                u_hur.norm(),
                                delta_p.norm(),
                            ]
                        ).detach().float().cpu()
                        f1, f2, f3, hur_norm, controlled_norm = map(
                            float, t5
                        )
                    else:
                        f1 = float(S_vec.pow(2).sum().item())
                        f2 = float(C_vec.pow(2).sum().item())
                        f3 = float(N_vec.pow(2).sum().item())
                        hur_norm = float(u_hur.norm().item())
                        controlled_norm = float(delta_p.norm().item())

        self.subspace.append(raw)
        if self.anchor_lambda > 0:
            p0 = self._init_snap.vector.to(self.device)
            p_curr = self.prompt.vector()
            applied = applied + self.anchor_lambda * (p0 - p_curr)
        ctrl_pre_anchor = controlled_norm
        self.prompt.apply_update(applied)
        self.step_idx += 1

        if stable and not rollback:
            self._save_checkpoint()

        applied_final = applied
        info = self._make_step_info(
            raw,
            applied_final,
            spec20,
            sv_list,
            rho=rho,
            scale=scale,
            rollback=False,
            q_mean=q_mean,
            p_mean=p_mean,
            f1=f1,
            f2=f2,
            f3=f3,
            state=state,
            is_warmup=is_warmup,
            projected_norm=projected_norm,
            off_norm=off_norm,
            u_raw_n=u_raw_n,
            hur_norm=hur_norm,
            controlled_norm=ctrl_pre_anchor,
            Q=Q,
            y_curr=y_curr,
            y_prev=y_prev,
            e=e,
            a_coeff=a_coeff,
            decomp=decomp,
            S_vec=S_vec,
            C_vec=C_vec,
            N_vec=N_vec,
            u_hur=u_hur,
            delta_p=delta_p,
        )
        return info

    def adapt_and_predict(
        self,
        x: torch.Tensor,
        loss_fn=None,
        return_all_step_infos: bool = False,
    ):
        """Run K adaptation steps on input x, then predict.

        Returns
        -------
        preds, last_info
            When ``return_all_step_infos`` is False (default).
        preds, last_info, step_infos
            When True: ``step_infos`` is a list of length K with one
            :class:`TPDStepInfo` per inner adaptation step.
        """
        if loss_fn is None:

            def loss_fn(logits):
                probs = torch.softmax(logits, dim=-1)
                return -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()

        last_info: Optional[TPDStepInfo] = None
        step_infos: List[TPDStepInfo] = []

        for _ in range(self.K):
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = loss_fn(logits)
            loss.backward()
            last_info = self._step()
            if return_all_step_infos and last_info is not None:
                step_infos.append(last_info)

        with torch.no_grad():
            logits = self.model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

        preds = logits.argmax(dim=-1)
        if return_all_step_infos:
            return preds, last_info, step_infos
        return preds, last_info
