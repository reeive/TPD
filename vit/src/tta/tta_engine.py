"""
TTA Engine: Closed-loop test-time adaptation with Koopman + HBUO.

Orchestrates the C1 (spectral stability) and C2 (effective direction)
controllers to perform test-time prompt adaptation on a pre-trained
VPT model via entropy minimization.

Optional TTA additions:
  T1 — Koopman-Regularized Head Adaptation (KRHA)
  T2 — Online Feature Prototypes (OFP-DR)
  T3 — Multi-Objective Loss (attention entropy + pseudo-labels)
  T4 — Koopman-SAM (sharpness along unstable dynamical modes)
  T5 — EMA Teacher-Student (Lyapunov stability anchor)
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from .koopman_controller import KoopmanController
from .hbuo import HBUO
from .peft_hbuo import PEFTNativeHBUO
from .state_utils import PromptStateManager
from .kdmd import KernelLifter, ClassKoopmanAccumulator, KDMDPredictor
from .ktmv import KoopmanTrajectoryMultiView
from .prototypes import OnlinePrototypeBank


class TTAEngine:
    """Test-Time Adaptation engine with Koopman + HBUO control."""

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: Optional[torch.device] = None,
    ):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.cfg = cfg

        self.protocol = cfg.get("protocol", "online")

        tta_cfg = cfg.get("tta", {})
        state_cfg = tta_cfg.get("state", {})
        koopman_cfg = tta_cfg.get("koopman", {})
        hbuo_cfg = tta_cfg.get("hbuo", {})
        update_cfg = tta_cfg.get("update", {})

        self.base_lr = update_cfg.get("lr", 1e-3)
        self.steps_per_sample = update_cfg.get("steps_per_sample", 1)
        self.selection_p = update_cfg.get("selection_p", 0.1)

        self.confidence_threshold = update_cfg.get("confidence_threshold", 0.0)
        self.ensemble_alpha = update_cfg.get("ensemble_alpha", 0.0)
        state_dim = state_cfg.get("dim", 16)
        projection_type = state_cfg.get("projection", "pca")

        # ---- Core components ----
        adaptive_r = state_cfg.get("adaptive_r", False)
        max_state_dim = state_cfg.get("max_state_dim", 512)
        self.state_manager = PromptStateManager(
            model=self.model,
            state_dim=state_dim,
            projection_type=projection_type,
            device=self.device,
            adaptive_r=adaptive_r,
            max_state_dim=max_state_dim,
        )

        self.koopman = KoopmanController(
            base_lr=self.base_lr,
            window_size=koopman_cfg.get("window", 10),
            ridge_lambda=koopman_cfg.get("ridge_lambda", 0.01),
            rollback_patience=koopman_cfg.get("rollback_patience", 3),
            state_dim=state_dim,
            min_lr_ratio=koopman_cfg.get("min_lr_ratio", 0.1),
            max_lr_ratio=koopman_cfg.get("max_lr_ratio", 2.0),
            rho_threshold=koopman_cfg.get("rho_threshold", 1.0),
            c1_mode=koopman_cfg.get("c1_mode", "full"),
            device=self.device,
            auto_cond_threshold=koopman_cfg.get("auto_cond_threshold", 1e8),
        )
        gar_target = koopman_cfg.get("gar_target", 0.5)
        self.koopman._gar_target = gar_target

        self.single_view = cfg.get("single_view", False)

        self.hbuo = HBUO(
            state_dim=state_dim,
            num_perturbations=hbuo_cfg.get("num_perturbations", 5),
            perturbation_scale=hbuo_cfg.get("perturbation_scale", 0.01),
            cross_episode=self.single_view,
            device=self.device,
        )
        self.hbuo.c2_variant = hbuo_cfg.get("variant", "uncentered")
        self.hbuo.c2_gate_threshold = hbuo_cfg.get("gate_threshold", 0.85)
        self.hbuo.c2_gate_blend = hbuo_cfg.get("gate_blend", 0.5)
        self.hbuo.c2_shrink_ratio = hbuo_cfg.get("shrink_ratio", 0.5)

        self.c2_mode = hbuo_cfg.get("mode", "fast")
        self.c2_energy_threshold = hbuo_cfg.get("energy_threshold", 0.95)

        self.gramian_update_freq = hbuo_cfg.get("update_freq", 10)
        self._sample_count = 0

        # ---- A-soft C2: EMA direction soft blend ----
        if self.c2_mode == "a_soft":
            asoft_cfg = hbuo_cfg.get("a_soft", {})
            self._asoft_mu: Optional[torch.Tensor] = None
            self._asoft_beta = asoft_cfg.get("beta", 0.9)
            self._asoft_alpha_base = asoft_cfg.get("alpha_base", 0.0)
            self._asoft_alpha_slope = asoft_cfg.get("alpha_slope", 0.3)
            self._asoft_snr_ref = asoft_cfg.get("snr_ref", 1.0)
            self._asoft_alpha_max = asoft_cfg.get("alpha_max", 0.6)
            self._asoft_last_sub_grads: Optional[torch.Tensor] = None

        # ---- SNR/consensus gate ----
        snr_gate_cfg = hbuo_cfg.get("snr_gate", {})
        self._snr_gate_enabled = snr_gate_cfg.get("enabled", False)
        self._snr_gate_tau_snr = snr_gate_cfg.get("tau_snr", 0.7)
        self._snr_gate_tau_cos = snr_gate_cfg.get("tau_cos", 0.15)
        self._snr_gate_steepness = snr_gate_cfg.get("steepness", 5.0)
        self._snr_gate_mu: Optional[torch.Tensor] = None
        self._snr_gate_mu_beta = snr_gate_cfg.get("mu_beta", 0.9)

        # ---- Koopman-Hankel Risk Gate ----
        kh_cfg = hbuo_cfg.get("kh_gate", {})
        self._kh_gate_enabled = kh_cfg.get("enabled", False)
        self._kh_delay_L = kh_cfg.get("delay_L", 2)
        self._kh_ridge = kh_cfg.get("ridge", 1e-3)
        self._kh_window = kh_cfg.get("window", 15)
        self._kh_gate_scale = kh_cfg.get("gate_scale", 5.0)
        self._kh_gate_threshold = kh_cfg.get("gate_threshold", 0.3)
        self._kh_risk_history: List[np.ndarray] = []
        self._kh_K_matrix: Optional[np.ndarray] = None
        self._kh_grad_mu: Optional[torch.Tensor] = None
        self._kh_grad_mu_beta = kh_cfg.get("grad_mu_beta", 0.9)
        self._kh_prev_loss = None
        self._kh_prev_conf = None

        # ---- Spectral Bridge C2 (c2_mode = "spectral_bridge") ----
        sb_cfg = hbuo_cfg.get("spectral_bridge", {})
        self._sb_enabled = (self.c2_mode == "spectral_bridge")
        self._sb_beta = sb_cfg.get("beta", 0.85)
        self._sb_lambda_rK = sb_cfg.get("lambda_rK", 0.5)
        self._sb_gamma = sb_cfg.get("gamma", 1.0)
        self._sb_w_min = sb_cfg.get("w_min", 0.05)
        self._sb_mu: Optional[np.ndarray] = None
        self._sb_sigma2: Optional[np.ndarray] = None
        self._sb_prev_y: Optional[np.ndarray] = None
        self._sb_prev_a: Optional[np.ndarray] = None

        # ---- Conflict-Spectral C2 (c2_mode = "conflict_spectral") ----
        cs_cfg = hbuo_cfg.get("conflict_spectral", {})
        self._cs_enabled = self.c2_mode in ("conflict_spectral", "conflict_router")
        self._cs_beta_ema = cs_cfg.get("beta_ema", 0.85)
        self._cs_flip_L = cs_cfg.get("flip_L", 5)
        self._cs_alpha = cs_cfg.get("alpha", 1.0)
        self._cs_beta_n = cs_cfg.get("beta_n", 0.3)
        self._cs_mu: Optional[np.ndarray] = None
        self._cs_sigma2: Optional[np.ndarray] = None
        self._cs_y_history: List[np.ndarray] = []

        # ---- Conflict-Router C2 (c2_mode = "conflict_router") ----
        cr_cfg = hbuo_cfg.get("conflict_router", {})
        self._cr_enabled = (self.c2_mode == "conflict_router")
        self._cr_tau_s = cr_cfg.get("tau_s", 3.0)
        self._cr_tau_c = cr_cfg.get("tau_c", 3.0)
        self._cr_tau_n = cr_cfg.get("tau_n", 3.0)
        self._cr_buffer_decay = cr_cfg.get("buffer_decay", 0.9)
        self._cr_buffer_weight = cr_cfg.get("buffer_weight", 0.3)
        self._cr_buffer: Optional[torch.Tensor] = None

        # ---- q/p decomposition C2 (c2_mode = "qp_decomp") ----
        qp_cfg = hbuo_cfg.get("qp_decomp", {})
        self._qp_kappa = qp_cfg.get("kappa", 2.0)
        self._qp_eps = qp_cfg.get("eps", 1e-6)
        self._qp_buffer_decay = qp_cfg.get(
            "buffer_decay", cr_cfg.get("buffer_decay", 0.9))
        self._qp_buffer_weight = qp_cfg.get(
            "buffer_weight", cr_cfg.get("buffer_weight", 0.3))
        self._qp_y_prev: Optional[torch.Tensor] = None
        self._qp_buffer: Optional[torch.Tensor] = None

        # ---- PEFT-native HBUO (c2_mode = "hbuo_peft") ----
        if self.c2_mode == "hbuo_peft":
            peft_cfg = hbuo_cfg.get("peft", {})
            self.peft_hbuo = PEFTNativeHBUO(
                prompt_basis_dim=peft_cfg.get("basis_dim", state_dim),
                hankel_len=peft_cfg.get("hankel_len", 5),
                horizon=peft_cfg.get("horizon", 3),
                shrink_ratio=hbuo_cfg.get("shrink_ratio", 0.5),
                energy_threshold=self.c2_energy_threshold,
                update_freq=peft_cfg.get("update_freq", 10),
                warmup=peft_cfg.get("warmup", 30),
                gram_horizon=peft_cfg.get("gram_horizon", 20),
                max_history=200,
                cross_episode=self.single_view,
                device=self.device,
            )
        else:
            self.peft_hbuo = None

        # ---- Koopman-Balanced C2 (c2_mode = "koopman_bal") ----
        if self.c2_mode == "koopman_bal":
            kbal_cfg = hbuo_cfg.get("koopman_bal", {})
            self._kbal_mu_hat: Optional[torch.Tensor] = None
            self._kbal_mu_beta = kbal_cfg.get("mu_beta", 0.9)
            self._kbal_Pi_bal: Optional[torch.Tensor] = None
            self._kbal_update_freq = kbal_cfg.get("update_freq", 5)
            self._kbal_gram_horizon = kbal_cfg.get("gram_horizon", 15)
            self._kbal_energy = hbuo_cfg.get("energy_threshold", 0.95)
            self._kbal_step_count = 0
            self._kbal_warmup = kbal_cfg.get("warmup", 3)
            self._kbal_last_hsv: Optional[torch.Tensor] = None
            self._kbal_bal_rank: int = 0
            # Force-aware adaptive shrinkage (F2/F1 driven)
            self._kbal_shrink_min = kbal_cfg.get("shrink_min", 0.05)
            self._kbal_shrink_max = kbal_cfg.get("shrink_max", 0.5)
            self._kbal_fa_tau = kbal_cfg.get("fa_tau", 5.0)
            self._kbal_fa_eps = 1e-8
            self._kbal_last_ratio = 0.0

        # ---- Three-Force-Aware Adaptive C2 (c2_mode = "force_adaptive") ----
        if self.c2_mode == "force_adaptive":
            fa_cfg = hbuo_cfg.get("force_adaptive", {})
            self._fa_mu_hat: Optional[torch.Tensor] = None
            self._fa_mu_beta = fa_cfg.get("mu_beta", 0.9)
            self._fa_eps = 1e-8
            self._fa_shrink_min = fa_cfg.get("shrink_min", 0.05)
            self._fa_shrink_max = fa_cfg.get("shrink_max", 0.8)
            self._fa_energy_min = fa_cfg.get("energy_min", 0.7)
            self._fa_energy_max = fa_cfg.get("energy_max", 0.99)
            self._fa_tau = fa_cfg.get("tau", 5.0)
            self._fa_signal_proj_threshold = fa_cfg.get(
                "signal_proj_threshold", 20.0)
            self._fa_warmup = fa_cfg.get("warmup", 5)
            self._fa_step_count = 0
            self._fa_last_ratio = 0.0

        # ---- Hankel-FA C2 (c2_mode = "hankel_fa") ----
        if self.c2_mode == "hankel_fa":
            hfa_cfg = hbuo_cfg.get("hankel_fa", {})
            self._hfa_mu_hat: Optional[torch.Tensor] = None
            self._hfa_mu_beta = hfa_cfg.get("mu_beta", 0.9)
            self._hfa_eps = 1e-8
            self._hfa_sp_threshold = hfa_cfg.get(
                "signal_proj_threshold", 10.0)
            self._hfa_warmup = hfa_cfg.get("warmup", 5)
            self._hfa_step = 0
            self._hfa_last_ratio = 0.0
            self._hfa_hankel_len = hfa_cfg.get("hankel_len", 15)
            self._hfa_hankel_energy = hfa_cfg.get("hankel_energy", 0.90)
            self._hfa_update_freq = hfa_cfg.get("update_freq", 5)
            self._hfa_shrink_noise = hfa_cfg.get("shrink_noise", 0.05)
            self._hfa_shrink_moderate = hfa_cfg.get("shrink_moderate", 0.5)
            self._hfa_grad_history: list = []
            self._hfa_Pi_hankel: Optional[torch.Tensor] = None
            self._hfa_hankel_rank: int = 0

        # ---- Gradient reuse mode (Scheme B) ----
        self.grad_reuse = update_cfg.get("grad_reuse", False)
        self.koopman_grad_transport = update_cfg.get(
            "koopman_grad_transport", False)

        # ---- View sub-batch mode ----
        self.view_subbatch = update_cfg.get("view_subbatch", False)
        self.n_view_groups = update_cfg.get("n_view_groups", 4)
        self.vsga_mode = update_cfg.get("vsga_mode", "denoise")

        # ---- Online drift protection ----
        self.max_prompt_drift = update_cfg.get("max_prompt_drift", 0.0)

        # ---- Anchor regularization: λ * ||P - P₀||² ----
        anchor_cfg = tta_cfg.get("anchor", {})
        self.anchor_lambda = anchor_cfg.get("lambda", 0.0)
        self._anchor_prompt_state: Optional[Dict[str, torch.Tensor]] = None

        # ---- Optimizer ----
        optim_cfg = update_cfg.get("optimizer", {})
        self.optimizer_type = optim_cfg.get("type", "sgd")
        self.use_scaler = optim_cfg.get("grad_scaler", False)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self._initial_optim_state: Optional[dict] = None

        # ---- T1: Head Adaptation (KRHA) ----
        scope_cfg = tta_cfg.get("scope", {})
        self.tune_head = scope_cfg.get("tune_head", False)
        self.tune_ln = scope_cfg.get("tune_ln", False)
        self.head_lr_mult = scope_cfg.get("head_lr_mult", 0.01)
        self.head_grad_clip = scope_cfg.get("head_grad_clip", 1.0)

        self._setup_grad_flags()
        self._head_param_names: List[str] = []
        self._cache_head_param_refs()

        # Stable state for rollback (prompt + head)
        self._stable_prompt_state: Optional[Dict[str, torch.Tensor]] = None
        self._stable_head_state: Optional[Dict[str, torch.Tensor]] = None
        self._head_init_state: Optional[Dict[str, torch.Tensor]] = None
        self._save_stable_state()
        if self.tune_head:
            self._head_init_state = {
                n: p.data.clone() for n, p in self._iter_head_params()
            }

        # Build optimizer after grad flags are set
        self._build_optimizer()

        # ---- Anchor: save initial prompt state ----
        if self.anchor_lambda > 0:
            prompt_params = self.state_manager.get_prompt_params(self.model)
            self._anchor_prompt_state = {
                i: p.data.clone() for i, p in enumerate(prompt_params)
            }

        # ---- T2: Feature Prototypes (OFP-DR) ----
        proto_cfg = tta_cfg.get("prototypes", {})
        self.proto_enabled = proto_cfg.get("enabled", False)
        self.proto_bank: Optional[OnlinePrototypeBank] = None
        if self.proto_enabled:
            num_classes = cfg.get("num_classes", 200)
            feat_dim = proto_cfg.get("feat_dim", 768)
            self.proto_bank = OnlinePrototypeBank(
                num_classes=num_classes,
                feat_dim=feat_dim,
                gamma=proto_cfg.get("gamma", 1.0),
                tau=proto_cfg.get("tau", 0.1),
                momentum=proto_cfg.get("momentum", 0.9),
                conf_threshold=proto_cfg.get("conf_threshold", 0.5),
                device=self.device,
            )

        # ---- T3: Multi-Objective Loss ----
        loss_cfg = tta_cfg.get("loss", {})
        self.lambda_attn = loss_cfg.get("lambda_attn", 0.0)
        self.lambda_pl = loss_cfg.get("lambda_pl", 0.0)
        self.pl_threshold = loss_cfg.get("pl_threshold", 0.7)

        if self.lambda_attn > 0:
            self._enable_attention_collection()

        # ---- T4: Koopman-SAM ----
        sam_cfg = tta_cfg.get("sam", {})
        self.sam_enabled = sam_cfg.get("enabled", False)
        self.sam_rho = sam_cfg.get("rho", 0.05)

        # ---- T5: EMA Teacher-Student ----
        ema_cfg = tta_cfg.get("ema", {})
        self.ema_enabled = ema_cfg.get("enabled", False)
        self.ema_alpha = ema_cfg.get("alpha", 0.999)
        self.ema_temperature = ema_cfg.get("temperature", 1.0)
        self._ema_warmup = ema_cfg.get("warmup", 5)
        self._ema_params: Optional[Dict[str, torch.Tensor]] = None
        if self.ema_enabled:
            self._init_ema()

        # ---- KTMV ----
        ktmv_cfg = tta_cfg.get("ktmv", {})
        self.ktmv_enabled = ktmv_cfg.get("enabled", False)
        self.ktmv: Optional[KoopmanTrajectoryMultiView] = None
        if self.ktmv_enabled:
            self.ktmv = KoopmanTrajectoryMultiView(
                n_views=ktmv_cfg.get("n_views", 4),
                perturbation_scale=ktmv_cfg.get("scale", 0.1),
                device=self.device,
            )

        # ---- KDMD ----
        kdmd_cfg = tta_cfg.get("kdmd", {})
        self.kdmd_enabled = kdmd_cfg.get("enabled", False)
        self.kdmd: Optional[KDMDPredictor] = None
        self.kernel_lifter: Optional[KernelLifter] = None
        if self.kdmd_enabled:
            num_classes = cfg.get("num_classes", 200)
            kdmd_dim = kdmd_cfg.get("dim", 128)
            kdmd_gamma = kdmd_cfg.get("gamma", 1.0)
            kdmd_lambda = kdmd_cfg.get("lam", 1.0)
            kdmd_temp = kdmd_cfg.get("temperature", 1.0)
            self.kernel_lifter = KernelLifter(
                input_dim=num_classes, output_dim=kdmd_dim,
                gamma=kdmd_gamma, device=self.device)
            accumulator = ClassKoopmanAccumulator(
                num_classes=num_classes, lifted_dim=kdmd_dim,
                ridge_lambda=0.01, device=self.device)
            self.kdmd = KDMDPredictor(
                accumulator=accumulator, lam=kdmd_lambda,
                temperature=kdmd_temp,
                min_class_coverage=0.3, min_samples_per_class=3)

        # ---- Metrics ----
        self.metrics: Dict[str, list] = {
            "entropy": [], "rho": [], "eta": [],
            "drift": [], "accuracy": [],
        }
        self._grad_history: List[np.ndarray] = []
        self._c2_proj_dim = state_dim

        # ---- Diagnostics (off by default, enable via collect_diagnostics) ----
        self.collect_diagnostics = False
        self.diagnostics_log: Dict[str, list] = {
            "rho_per_sample": [],
            "grad_sv_raw": [],       # singular values of raw gradient history
            "grad_sv_filtered": [],  # singular values after C2 filtering
            "grad_norm_raw": [],
            "grad_norm_filtered": [],
            "c2_effective_rank": [],
            "prediction_before": [],
            "prediction_after": [],
        }

    # ====================================================================
    # Initialization helpers
    # ====================================================================

    def _setup_grad_flags(self):
        """Freeze everything except selected parameters."""
        for name, param in self.model.named_parameters():
            if "prompt" in name and ("embeddings" in name or "ctx" in name):
                param.requires_grad = True
            elif "prompt_learner" in name:
                param.requires_grad = True
            elif self.tune_head and "head" in name:
                param.requires_grad = True
            elif self.tune_ln and ("norm" in name.lower() or "ln" in name.lower()):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _build_optimizer(self):
        """Build optimizer (AdamW or SGD) and optional GradScaler."""
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            return

        if self.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(trainable, lr=self.base_lr)
        elif self.optimizer_type == "sgd_momentum":
            self.optimizer = torch.optim.SGD(
                trainable, lr=self.base_lr, momentum=0.9)
        else:
            self.optimizer = None

        if self.use_scaler and self.optimizer is not None:
            self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        if self.optimizer is not None:
            self._initial_optim_state = copy.deepcopy(
                self.optimizer.state_dict())

    def _cache_head_param_refs(self):
        """Cache head parameter names for efficient access."""
        self._head_param_names = []
        for name, param in self.model.named_parameters():
            if "head" in name and param.requires_grad:
                self._head_param_names.append(name)

    def _iter_head_params(self):
        """Yield (name, param) for trainable head parameters."""
        for name, param in self.model.named_parameters():
            if "head" in name and param.requires_grad:
                yield name, param

    def _get_head_params(self) -> List[nn.Parameter]:
        """Return list of trainable head parameters."""
        return [p for _, p in self._iter_head_params()]

    def _get_all_trainable_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Return (prompt_params, head_params)."""
        prompt_params = self.state_manager.get_prompt_params(self.model)
        head_params = self._get_head_params() if self.tune_head else []
        return prompt_params, head_params

    def _enable_attention_collection(self):
        """Enable attention weight collection in the ViT backbone."""
        model = self.model
        enc = model.enc if hasattr(model, "enc") else model
        transformer = enc.transformer if hasattr(enc, "transformer") else enc
        if hasattr(transformer, "encoder"):
            transformer.encoder.vis = True
            if hasattr(transformer.encoder, "layer"):
                for layer in transformer.encoder.layer:
                    if hasattr(layer, "attn"):
                        layer.attn.vis = True

    # ====================================================================
    # EMA Teacher (T5)
    # ====================================================================

    def _init_ema(self):
        """Initialize EMA parameter copies from current trainable params."""
        self._ema_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._ema_params[name] = param.data.clone()

    @torch.no_grad()
    def _update_ema(self):
        """EMA update: ema_p = alpha * ema_p + (1-alpha) * p."""
        if self._ema_params is None:
            return
        alpha = self.ema_alpha
        for name, param in self.model.named_parameters():
            if name in self._ema_params:
                self._ema_params[name].mul_(alpha).add_(
                    param.data, alpha=1.0 - alpha)

    @torch.no_grad()
    def _ema_teacher_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get teacher logits by temporarily loading EMA params."""
        if self._ema_params is None:
            return self.model(x)

        originals = {}
        for name, param in self.model.named_parameters():
            if name in self._ema_params:
                originals[name] = param.data.clone()
                param.data.copy_(self._ema_params[name])

        self.model.eval()
        logits = self.model(x)

        for name, param in self.model.named_parameters():
            if name in originals:
                param.data.copy_(originals[name])

        return logits

    @property
    def _ema_ready(self) -> bool:
        return (self.ema_enabled
                and self._ema_params is not None
                and self._sample_count > self._ema_warmup)

    # ====================================================================
    # State management (rollback / episodic)
    # ====================================================================

    def _save_stable_state(self):
        """Cache current prompt (and head) state as stable checkpoint."""
        prompt_params = self.state_manager.get_prompt_params(self.model)
        self._stable_prompt_state = {
            i: p.data.clone() for i, p in enumerate(prompt_params)
        }
        if self.tune_head:
            self._stable_head_state = {
                n: p.data.clone() for n, p in self._iter_head_params()
            }

    def _rollback_to_stable(self):
        """Restore prompts (and head) to last stable cached state."""
        prompt_params = self.state_manager.get_prompt_params(self.model)
        for i, p in enumerate(prompt_params):
            p.data.copy_(self._stable_prompt_state[i])
        if self.tune_head and self._stable_head_state is not None:
            for name, param in self._iter_head_params():
                if name in self._stable_head_state:
                    param.data.copy_(self._stable_head_state[name])

    def _reset_for_episodic(self):
        """Reset prompt to P_0 and clear all state (episodic mode)."""
        prompt_params = self.state_manager.get_prompt_params(self.model)
        offset = 0
        for p in prompt_params:
            numel = p.numel()
            p.data.copy_(
                self.state_manager.mean[offset:offset + numel].reshape(p.shape)
            )
            offset += numel

        if self.tune_head and self._head_init_state is not None:
            for name, param in self._iter_head_params():
                if name in self._head_init_state:
                    param.data.copy_(self._head_init_state[name])

        if self.optimizer is not None and self._initial_optim_state is not None:
            self.optimizer.load_state_dict(
                copy.deepcopy(self._initial_optim_state))

        self.state_manager.reset_buffer()
        self.koopman.reset()
        self.hbuo.reset()
        if self.ktmv is not None:
            self.ktmv.reset()
        if self.proto_bank is not None:
            self.proto_bank.reset()
        self._qp_y_prev = None
        self._qp_buffer = None
        if self.ema_enabled:
            self._init_ema()
        self._save_stable_state()

    # ====================================================================
    # Loss computation
    # ====================================================================

    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Entropy loss (marginal for multi-view, standard for single)."""
        if logits.shape[0] > 1:
            avg_probs = F.softmax(logits, dim=-1).mean(dim=0)
            return -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
        else:
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            return -(probs * log_probs).sum(dim=-1).mean()

    def _attention_entropy_loss(
        self, attn_weights: List[Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Attention entropy of CLS-to-patch attention in last layer.

        Encourages the ViT to maintain focused attention under
        distribution shift (LookSharp, NeurIPS 2025).
        """
        last_attn = attn_weights[-1] if attn_weights else None
        if last_attn is None:
            return torch.tensor(0.0, device=self.device)

        # last_attn: (B, num_heads, seq_len, seq_len)
        # CLS token attention over all tokens: row 0
        cls_attn = last_attn[:, :, 0, :]  # (B, H, S)
        cls_attn = cls_attn.clamp(min=1e-8)
        entropy = -(cls_attn * cls_attn.log()).sum(dim=-1)  # (B, H)
        return entropy.mean()

    def _pseudo_label_loss(
        self, student_logits: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence from EMA teacher predictions (soft pseudo-labels)."""
        teacher_logits = self._ema_teacher_forward(x)
        teacher_probs = F.softmax(
            teacher_logits / self.ema_temperature, dim=-1)
        max_prob = teacher_probs.max(dim=-1).values.mean()
        if max_prob < self.pl_threshold:
            return torch.tensor(0.0, device=self.device)

        student_log_probs = F.log_softmax(
            student_logits / self.ema_temperature, dim=-1)
        T2 = self.ema_temperature ** 2
        return F.kl_div(
            student_log_probs, teacher_probs.detach(),
            reduction="batchmean") * T2

    def _compute_loss(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """Forward pass + multi-objective loss.

        Returns:
            (loss, attn_weights_or_None)
        """
        attn_weights = None

        if self._ktmv_is_ready():
            logits_views = self._ktmv_forward(x)
            loss = KoopmanTrajectoryMultiView.compute_marginal_entropy(
                logits_views)
            student_logits = logits_views[0]
        else:
            self.model.train()
            if self.lambda_attn > 0:
                logits, attn_weights = self._forward_with_attn(x)
            else:
                logits = self.model(x)
            logits_conf = self._select_confident_samples(logits)
            loss = self._entropy_loss(logits_conf)
            student_logits = logits_conf

            if attn_weights and self.lambda_attn > 0:
                loss_attn = self._attention_entropy_loss(attn_weights)
                loss = loss + self.lambda_attn * loss_attn

        if self._ema_ready and self.lambda_pl > 0:
            loss_pl = self._pseudo_label_loss(student_logits, x)
            loss = loss + self.lambda_pl * loss_pl

        return loss, attn_weights

    def _compute_loss_cached(
        self, cached_image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """Forward pass using cached image features (text path only).

        Skips image encoder entirely; only runs text encoding (which depends
        on the tunable ctx parameter) and computes entropy loss.
        """
        self.model.train()
        logits = self.model.forward_from_cache(cached_image_features)
        logits_conf = self._select_confident_samples(logits)
        loss = self._entropy_loss(logits_conf)

        if self._ema_ready and self.lambda_pl > 0:
            loss_pl = self._pseudo_label_loss(logits_conf, None)
            loss = loss + self.lambda_pl * loss_pl

        return loss, None

    # ====================================================================
    # C2 gradient filtering
    # ====================================================================

    # ====================================================================
    # Koopman-Balanced C2
    # ====================================================================

    def _update_koopman_bal_projector(self):
        """Recompute balanced truncation projector from C1's Koopman operator.

        Uses A_t (already fitted by C1) and mu_hat (gradient signal EMA)
        to compute controllability and observability Gramians, then performs
        balanced truncation to find directions that are simultaneously
        dynamically excitable and aligned with the adaptation signal.

        Supports both full and diagonal C1 modes: for diagonal mode,
        constructs A = diag(a_1, ..., a_r) from per-dimension growth rates.
        """
        A = self.koopman._last_A_hat
        if A is None and self.koopman._last_diag_a is not None:
            A = torch.diag(self.koopman._last_diag_a)
        if A is None:
            return
        mu = self._kbal_mu_hat
        if mu is None or mu.norm() < 1e-12:
            return

        r = A.shape[0]
        dev = A.device
        K = self._kbal_gram_horizon

        # C = normalized signal direction (connects to Three-Force F1)
        c = (mu / mu.norm()).unsqueeze(0)  # (1, r)
        CtC = c.T @ c  # (r, r), rank-1

        W_c = torch.zeros(r, r, device=dev)
        W_o = torch.zeros(r, r, device=dev)
        Ak = torch.eye(r, device=dev)
        for _ in range(K):
            W_c += Ak @ Ak.T
            W_o += Ak.T @ CtC @ Ak
            Ak = A @ Ak
            if Ak.abs().max() > 1e6:
                break

        reg = 1e-6 * torch.eye(r, device=dev)
        W_c = W_c + reg
        W_o = W_o + reg

        WcWo = W_c @ W_o
        WcWo_sym = 0.5 * (WcWo + WcWo.T)

        try:
            eigvals, eigvecs = torch.linalg.eigh(WcWo_sym)
        except RuntimeError:
            return

        hsv = eigvals.clamp(min=0).sqrt()
        idx = hsv.argsort(descending=True)
        hsv = hsv[idx]
        eigvecs = eigvecs[:, idx]
        self._kbal_last_hsv = hsv.detach()

        total = hsv.sum().item()
        if total < 1e-12:
            return

        cumsum = hsv.cumsum(0)
        k_arr = (cumsum >= self._kbal_energy * total).nonzero(as_tuple=True)[0]
        k_eff = max(1, k_arr[0].item() + 1) if len(k_arr) > 0 else r
        k_eff = min(k_eff, r)
        self._kbal_bal_rank = k_eff

        V_k = eigvecs[:, :k_eff]
        self._kbal_Pi_bal = V_k @ V_k.T

    def _filter_gradient_koopman_bal(self, grad_vec: torch.Tensor) -> torch.Tensor:
        """Filter gradient via Koopman-informed balanced truncation.

        Direction selection: balanced truncation of A_t (controllable + observable).
        Shrinkage intensity: driven by F2/F1 ratio (three-force aware).
          high F2/F1 → small shrink → aggressive denoising
          low  F2/F1 → large shrink → preserve gradient
        """
        U = self.state_manager.projection
        if U is None:
            return self.hbuo.filter_gradient_fast(
                grad_vec, energy_threshold=self.c2_energy_threshold)

        g_z = U.T @ grad_vec  # (r,)

        if self._kbal_mu_hat is None:
            self._kbal_mu_hat = g_z.detach().clone()
        else:
            beta = self._kbal_mu_beta
            self._kbal_mu_hat = beta * self._kbal_mu_hat + (1 - beta) * g_z.detach()

        self._kbal_step_count += 1

        if self._kbal_step_count < self._kbal_warmup:
            return self.hbuo.filter_gradient_fast(
                grad_vec, energy_threshold=self.c2_energy_threshold)

        if (self._kbal_step_count % self._kbal_update_freq == 0
                or self._kbal_Pi_bal is None):
            self._update_koopman_bal_projector()

        if self._kbal_Pi_bal is None:
            return self.hbuo.filter_gradient_fast(
                grad_vec, energy_threshold=self.c2_energy_threshold)

        # Compute F2/F1-driven adaptive shrinkage
        mu = self._kbal_mu_hat
        mu_norm = mu.norm().item() + self._kbal_fa_eps
        f1 = torch.dot(g_z, mu).abs().item() / mu_norm
        f2 = (g_z - mu).norm().item()
        ratio = f2 / (f1 + self._kbal_fa_eps)
        self._kbal_last_ratio = ratio

        alpha = min(ratio / self._kbal_fa_tau, 1.0)
        shrink_t = self._kbal_shrink_max - alpha * (
            self._kbal_shrink_max - self._kbal_shrink_min)

        # Balanced filter in state space
        Pi = self._kbal_Pi_bal
        g_proj = Pi @ g_z
        g_z_filtered = g_proj + shrink_t * (g_z - g_proj)

        # Reconstruct: state-space filtered + shrunk residual
        g_basis = U @ g_z
        g_residual = grad_vec - g_basis
        return U @ g_z_filtered + shrink_t * g_residual

    def _filter_gradient_force_adaptive(
        self, grad_vec: torch.Tensor
    ) -> torch.Tensor:
        """Three-Force-Aware Adaptive C2.

        Monitors the noise-to-signal ratio (F2/F1) online and adapts
        shrinkage and energy thresholds accordingly:
        - High F2/F1 → aggressive denoising (low shrink, low energy)
        - Low F2/F1  → preserve gradient (high shrink, high energy)
        - Extreme F2/F1 → project gradient onto signal direction μ̂
        """
        self._fa_step_count += 1

        if self._fa_mu_hat is None:
            self._fa_mu_hat = grad_vec.detach().clone()
        else:
            beta = self._fa_mu_beta
            self._fa_mu_hat = (
                beta * self._fa_mu_hat + (1 - beta) * grad_vec.detach())

        if self._fa_step_count < self._fa_warmup:
            return self.hbuo.filter_gradient_fast(
                grad_vec, energy_threshold=self.c2_energy_threshold)

        mu = self._fa_mu_hat
        mu_norm = mu.norm().item() + self._fa_eps

        f1 = torch.dot(grad_vec, mu).item() / mu_norm
        f2 = (grad_vec - mu).norm().item()
        ratio = abs(f2) / (abs(f1) + self._fa_eps)
        self._fa_last_ratio = ratio

        if ratio > self._fa_signal_proj_threshold and mu_norm > self._fa_eps:
            mu_dir = mu / mu.norm()
            proj_coeff = torch.dot(grad_vec, mu_dir)
            return proj_coeff * mu_dir

        alpha = min(ratio / self._fa_tau, 1.0)
        shrink_t = self._fa_shrink_max - alpha * (
            self._fa_shrink_max - self._fa_shrink_min)
        energy_t = self._fa_energy_max - alpha * (
            self._fa_energy_max - self._fa_energy_min)

        orig_shrink = self.hbuo.c2_shrink_ratio
        self.hbuo.c2_shrink_ratio = shrink_t
        g_filtered = self.hbuo.filter_gradient_fast(
            grad_vec, energy_threshold=energy_t)
        self.hbuo.c2_shrink_ratio = orig_shrink

        return g_filtered

    def _filter_gradient_hankel_fa(
        self, grad_vec: torch.Tensor
    ) -> torch.Tensor:
        """Hankel + Force-Aware C2.

        Builds a block-Hankel matrix from the gradient history in state
        space, extracts temporally persistent signal modes via SVD, then
        uses the Three-Force ratio to decide filtering aggressiveness:
          - High F2/F1 → project onto Hankel signal subspace (rank-k)
          - Moderate F2/F1 → standard HAUO with shrinkage
        """
        self._hfa_step += 1
        U = self.state_manager.projection

        if U is not None:
            g_z = U.T @ grad_vec
        else:
            g_z = grad_vec

        self._hfa_grad_history.append(g_z.detach().clone())
        max_keep = self._hfa_hankel_len * 2
        if len(self._hfa_grad_history) > max_keep:
            self._hfa_grad_history = self._hfa_grad_history[-max_keep:]

        if self._hfa_mu_hat is None:
            self._hfa_mu_hat = g_z.detach().clone()
        else:
            beta = self._hfa_mu_beta
            self._hfa_mu_hat = (
                beta * self._hfa_mu_hat + (1 - beta) * g_z.detach())

        if self._hfa_step < self._hfa_warmup:
            return self.hbuo.filter_gradient_fast(
                grad_vec, energy_threshold=self.c2_energy_threshold)

        mu = self._hfa_mu_hat
        mu_norm = mu.norm().item() + self._hfa_eps
        f1 = torch.dot(g_z, mu).item() / mu_norm
        f2 = (g_z - mu).norm().item()
        ratio = abs(f2) / (abs(f1) + self._hfa_eps)
        self._hfa_last_ratio = ratio

        L = len(self._hfa_grad_history)
        if (L >= self._hfa_hankel_len
                and (self._hfa_step % self._hfa_update_freq == 0
                     or self._hfa_Pi_hankel is None)):
            self._update_hankel_projector()

        if ratio > self._hfa_sp_threshold:
            if self._hfa_Pi_hankel is not None:
                g_proj = self._hfa_Pi_hankel @ g_z
                g_z_filtered = (
                    g_proj + self._hfa_shrink_noise * (g_z - g_proj))
                if U is not None:
                    g_basis = U @ g_z
                    g_residual = grad_vec - g_basis
                    return (U @ g_z_filtered
                            + self._hfa_shrink_noise * g_residual)
                else:
                    return g_z_filtered
            else:
                mu_dir = self._hfa_mu_hat / (
                    self._hfa_mu_hat.norm() + self._hfa_eps)
                proj_c = torch.dot(g_z, mu_dir)
                g_z_filtered = proj_c * mu_dir
                if U is not None:
                    g_basis = U @ g_z
                    g_residual = grad_vec - g_basis
                    return (U @ g_z_filtered
                            + self._hfa_shrink_noise * g_residual)
                return g_z_filtered

        return self.hbuo.filter_gradient_fast(
            grad_vec, energy_threshold=self.c2_energy_threshold)

    def _update_hankel_projector(self):
        """Build block-Hankel matrix from gradient history in state
        space, SVD to find temporally persistent signal modes, and
        construct the Hankel signal subspace projector."""
        grads = self._hfa_grad_history[-self._hfa_hankel_len:]
        r = grads[0].shape[0]
        L = len(grads)

        p = L // 2
        q = L - p + 1
        if p < 2 or q < 2:
            return

        H = torch.zeros(
            p * r, q, device=grads[0].device, dtype=grads[0].dtype)
        for i in range(p):
            for j in range(q):
                idx = i + j
                if idx < L:
                    H[i * r:(i + 1) * r, j] = grads[idx]

        try:
            Uh, Sh, _ = torch.linalg.svd(H, full_matrices=False)
        except Exception:
            return

        if Sh.numel() == 0 or Sh[0].item() < self._hfa_eps:
            return

        cumsum = torch.cumsum(Sh ** 2, 0)
        total_e = cumsum[-1].item()
        if total_e < self._hfa_eps:
            return

        k_arr = (cumsum >= self._hfa_hankel_energy * total_e).nonzero(
            as_tuple=True)[0]
        k = k_arr[0].item() + 1 if k_arr.numel() > 0 else Sh.shape[0]
        k = max(1, min(k, r))
        self._hfa_hankel_rank = k

        Uk = Uh[:, :k].reshape(p, r, k)
        U_avg = Uk.mean(dim=0)

        try:
            Q, _ = torch.linalg.qr(U_avg)
        except Exception:
            return

        k_final = min(k, Q.shape[1])
        Pi = Q[:, :k_final] @ Q[:, :k_final].T
        self._hfa_Pi_hankel = Pi

    def _filter_gradient_diag_shrink(self, grad_vec: torch.Tensor) -> torch.Tensor:
        """C2-diag_shrink: Per-dimension spectral shrinkage using diagonal Koopman.

        Uses C1's per-dimension growth rates a_i to scale gradients:
          - Unstable dims (|a_i| > threshold): shrink by threshold/|a_i|
          - Stable dims (|a_i| <= threshold): pass through unchanged
          - Residual (outside state space): global shrink by threshold/ρ

        This makes C2 a targeted version of C1: instead of globally
        scaling LR, it selectively dampens the unstable gradient directions.
        """
        diag_a = self.koopman._last_diag_a
        if diag_a is None:
            return grad_vec

        U = self.state_manager.projection
        if U is None:
            return grad_vec

        thr = self.koopman.rho_threshold

        g_z = U.T @ grad_vec                      # (r,)

        magnitudes = diag_a.abs()
        scale = torch.where(
            magnitudes > thr,
            thr / magnitudes.clamp(min=1e-8),      # shrink unstable dims
            torch.ones_like(magnitudes),            # preserve stable dims
        )
        g_z_filtered = g_z * scale                 # (r,)

        g_basis = U @ g_z
        g_residual = grad_vec - g_basis
        rho = magnitudes.max().item()
        residual_scale = min(1.0, thr / max(rho, 1e-8))

        return U @ g_z_filtered + residual_scale * g_residual

    def _compute_conflict_scores(self, y_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (s_i, c_i, n_i) — shared/conflict/noise per Koopman mode."""
        r = len(y_t)
        beta = self._cs_beta_ema

        if self._cs_mu is None:
            self._cs_mu = y_t.copy()
            self._cs_sigma2 = np.zeros(r)
            self._cs_y_history = [y_t.copy()]
            s_i = np.ones(r) * 0.5
            c_i = np.zeros(r)
            n_i = np.zeros(r) + 1e-4
            return s_i, c_i, n_i

        self._cs_mu = beta * self._cs_mu + (1 - beta) * y_t
        self._cs_sigma2 = beta * self._cs_sigma2 + (1 - beta) * (y_t - self._cs_mu) ** 2

        self._cs_y_history.append(y_t.copy())
        if len(self._cs_y_history) > self._cs_flip_L:
            self._cs_y_history.pop(0)

        mu_sq = self._cs_mu ** 2
        s_i = mu_sq / (mu_sq + self._cs_sigma2 + 1e-8)

        flip_rate = np.zeros(r)
        L = len(self._cs_y_history)
        if L >= 3:
            for k in range(1, L):
                flip_rate += (np.sign(self._cs_y_history[k]) !=
                              np.sign(self._cs_y_history[k - 1])).astype(float)
            flip_rate /= (L - 1)
        c_i = flip_rate

        n_i = np.zeros(r) + 1e-4
        if L >= 3:
            residual = y_t - 0.5 * self._cs_y_history[-2] - 0.3 * self._cs_y_history[-3]
            n_i = residual ** 2

        return s_i, c_i, n_i

    def _filter_gradient_conflict_spectral(self, grad_vec: torch.Tensor) -> torch.Tensor:
        """Conflict-Spectral C2: suppress oscillatory conflict modes, preserve persistent.

        w_i = s_i / (s_i + α·c_i + β·n_i + ε)
        g_filtered = U · diag(w) · U^T · g  +  residual
        """
        U = self.state_manager.projection
        y_t = (U.T @ grad_vec.detach()).cpu().numpy()

        s_i, c_i, n_i = self._compute_conflict_scores(y_t)
        w = s_i / (s_i + self._cs_alpha * c_i + self._cs_beta_n * n_i + 1e-8)
        w = np.clip(w, 0.05, 1.0)

        w_tensor = torch.from_numpy(w).float().to(grad_vec.device)
        y_tensor = U.T @ grad_vec
        y_weighted = w_tensor * y_tensor
        g_filtered = U @ y_weighted

        g_in_U = U @ (U.T @ grad_vec)
        g_residual = grad_vec - g_in_U
        return g_filtered + g_residual

    def _filter_gradient_conflict_router(self, grad_vec: torch.Tensor) -> torch.Tensor:
        """Conflict-Router C2: route shared→global, conflict→buffer, noise→drop.

        Uses softmax routing per mode. The buffer accumulates conflict gradients
        and is blended back during the update, providing a side channel for
        class-conflicting directions.
        """
        U = self.state_manager.projection
        y_t_np = (U.T @ grad_vec.detach()).cpu().numpy()
        r = len(y_t_np)

        s_i, c_i, n_i = self._compute_conflict_scores(y_t_np)

        logits = np.stack([
            self._cr_tau_s * s_i,
            self._cr_tau_c * c_i,
            self._cr_tau_n * n_i,
        ], axis=0)
        logits -= logits.max(axis=0, keepdims=True)
        exp_l = np.exp(logits)
        pi = exp_l / (exp_l.sum(axis=0, keepdims=True) + 1e-10)

        pi_shared = torch.from_numpy(pi[0]).float().to(grad_vec.device)
        pi_conflict = torch.from_numpy(pi[1]).float().to(grad_vec.device)

        y_tensor = U.T @ grad_vec
        y_shared = pi_shared * y_tensor
        y_conflict = pi_conflict * y_tensor

        if self._cr_buffer is None:
            self._cr_buffer = torch.zeros(r, device=grad_vec.device)
        self._cr_buffer = self._cr_buffer * self._cr_buffer_decay + y_conflict.detach()

        y_out = y_shared + self._cr_buffer_weight * self._cr_buffer
        g_filtered = U @ y_out

        g_in_U = U @ (U.T @ grad_vec)
        g_residual = grad_vec - g_in_U
        return g_filtered + g_residual

    def _filter_gradient_qp_decomp(self, grad_vec: torch.Tensor) -> torch.Tensor:
        """C2: q/p decomposition from TPD derivation (Koopman a_i + buffer on c)."""
        U = self.state_manager.projection
        y_t = U.T @ grad_vec
        r = y_t.shape[0]
        diag_a = self.koopman._last_diag_a

        if diag_a is None or self._qp_y_prev is None:
            self._qp_y_prev = y_t.detach().clone()
            if self._qp_buffer is None:
                self._qp_buffer = torch.zeros(r, device=grad_vec.device)
            return grad_vec

        da = diag_a.detach().to(grad_vec.device)
        if da.shape[0] != r:
            self._qp_y_prev = y_t.detach().clone()
            return grad_vec

        y_prev = self._qp_y_prev.to(grad_vec.device)
        y_hat = da * y_prev
        residual = y_t - y_hat
        eps = self._qp_eps
        q = y_hat * y_hat / (y_hat * y_hat + residual * residual + eps)
        p = 0.5 * (1.0 + torch.tanh(self._qp_kappa * da))
        s = q * p * y_t
        c = q * (1.0 - p) * y_t

        if self._qp_buffer is None:
            self._qp_buffer = torch.zeros(r, device=grad_vec.device)
        self._qp_buffer = (
            self._qp_buffer * self._qp_buffer_decay + c.detach())

        y_out = s + self._qp_buffer_weight * self._qp_buffer
        g_filtered = U @ y_out
        g_in_U = U @ (U.T @ grad_vec)
        g_residual = grad_vec - g_in_U

        self._qp_y_prev = y_t.detach().clone()
        return g_filtered + g_residual

    def _filter_gradient_spectral_bridge(self, grad_vec: torch.Tensor) -> torch.Tensor:
        """Spectral Bridge C2: per-mode credibility weighting in Koopman coordinates.

        Computes credibility q_i = μ²/(μ² + σ² + λ·r_K² + ε) per Koopman mode,
        then applies w_i = clip(q_i^γ, w_min, 1) as a diagonal filter:
            g_filtered = U · diag(w) · U^T · g
        """
        U = self.state_manager.projection  # (d, r)
        y_t = (U.T @ grad_vec.detach()).cpu().numpy()  # (r,)
        r = len(y_t)

        # Initialize on first call
        if self._sb_mu is None:
            self._sb_mu = y_t.copy()
            self._sb_sigma2 = np.zeros(r)
            self._sb_prev_y = y_t.copy()
            self._sb_prev_a = np.ones(r)
            return grad_vec  # no filtering on first step

        # Koopman dynamics residual
        a_prev = self._sb_prev_a
        rK = y_t - a_prev * self._sb_prev_y

        # Update EMA statistics
        beta = self._sb_beta
        self._sb_mu = beta * self._sb_mu + (1 - beta) * y_t
        self._sb_sigma2 = beta * self._sb_sigma2 + (1 - beta) * (y_t - self._sb_mu) ** 2

        # Credibility spectrum
        mu_sq = self._sb_mu ** 2
        q = mu_sq / (mu_sq + self._sb_sigma2 + self._sb_lambda_rK * rK ** 2 + 1e-8)

        # Weights
        w = np.clip(q ** self._sb_gamma, self._sb_w_min, 1.0)

        # Store for next step
        self._sb_prev_y = y_t.copy()
        diag_a = self.koopman._last_diag_a
        if diag_a is not None:
            self._sb_prev_a = diag_a.detach().cpu().numpy()

        # Apply: g_filtered = U · diag(w) · U^T · g
        w_tensor = torch.from_numpy(w).float().to(grad_vec.device)
        y_tensor = U.T @ grad_vec  # (r,)
        y_weighted = w_tensor * y_tensor  # (r,)
        g_filtered = U @ y_weighted  # (d,)

        # Preserve residual (components outside U span)
        g_in_U = U @ (U.T @ grad_vec)
        g_residual = grad_vec - g_in_U
        return g_filtered + g_residual

    def _filter_gradient_a_soft(self, grad_vec: torch.Tensor) -> torch.Tensor:
        """A-soft C2: EMA direction soft blend in full gradient space.

        Maintains an EMA of gradient direction μ̂. Blends the current
        gradient toward μ̂ with adaptive strength α_t that increases
        when SNR is low (high noise → lean more on history).
        """
        if self._asoft_mu is None:
            self._asoft_mu = grad_vec.detach().clone()
        else:
            self._asoft_mu = (self._asoft_beta * self._asoft_mu
                              + (1 - self._asoft_beta) * grad_vec.detach())

        mu_norm = self._asoft_mu.norm()
        if mu_norm < 1e-10:
            return grad_vec

        mu_unit = self._asoft_mu / mu_norm
        g_proj = torch.dot(grad_vec, mu_unit) * mu_unit

        snr_est = self._estimate_snr_fast(grad_vec)
        alpha_t = (self._asoft_alpha_base
                   + self._asoft_alpha_slope * max(0, self._asoft_snr_ref - snr_est))
        alpha_t = min(alpha_t, self._asoft_alpha_max)

        return (1 - alpha_t) * grad_vec + alpha_t * g_proj

    def _estimate_snr_fast(self, grad_vec: torch.Tensor) -> float:
        """Quick SNR estimate using EMA vs current gradient divergence."""
        if self._asoft_mu is None:
            return 1.0
        cos = F.cosine_similarity(
            grad_vec.unsqueeze(0), self._asoft_mu.unsqueeze(0)).item()
        return max(0.01, (1 + cos) / 2 * 2)

    def compute_kh_gate(self, loss_t: float, conf_t: float,
                        rho_t: float, grad_vec: torch.Tensor) -> float:
        """Koopman-Hankel risk gate on low-dimensional risk state.

        State vector (7-dim, n_unstable removed per ablation):
          [loss, entropy, confidence, grad_cos, grad_norm, rho, delta_loss, delta_conf]

        Uses delay-2 Hankel embedding + ridge Koopman to predict next risk,
        then converts to a gate value via sigmoid.
        """
        if not self._kh_gate_enabled:
            return 1.0

        if self._kh_grad_mu is None:
            self._kh_grad_mu = grad_vec.detach().clone()
            grad_cos = 1.0
        else:
            grad_cos = F.cosine_similarity(
                grad_vec.unsqueeze(0), self._kh_grad_mu.unsqueeze(0)).item()
            self._kh_grad_mu = (self._kh_grad_mu_beta * self._kh_grad_mu
                                + (1 - self._kh_grad_mu_beta) * grad_vec.detach())

        delta_loss = (loss_t - self._kh_prev_loss) if self._kh_prev_loss is not None else 0.0
        delta_conf = (conf_t - self._kh_prev_conf) if self._kh_prev_conf is not None else 0.0

        x_t = np.array([
            loss_t,
            loss_t,          # entropy = loss for entropy-min
            conf_t,
            grad_cos,
            grad_vec.norm().item(),
            rho_t,
            delta_loss,
            delta_conf,
        ])

        self._kh_prev_loss = loss_t
        self._kh_prev_conf = conf_t

        # Predict before observing
        gate_val = 1.0
        if self._kh_K_matrix is not None and len(self._kh_risk_history) >= self._kh_delay_L:
            emb = np.concatenate([
                self._kh_risk_history[-1 - i]
                for i in range(self._kh_delay_L)
            ])
            pred = emb @ self._kh_K_matrix  # predicted next raw state

            pred_delta_loss = max(0, pred[0] - x_t[0])
            pred_delta_conf = max(0, x_t[2] - pred[2])
            pred_rho_rise = max(0, pred[5] - 1.0)
            pred_cos_drop = max(0, 0.2 - pred[3])

            risk_t = (0.3 * pred_delta_loss + 0.3 * pred_delta_conf
                      + 0.2 * pred_rho_rise + 0.2 * pred_cos_drop)

            gate_val = 1.0 / (1.0 + math.exp(
                self._kh_gate_scale * (risk_t - self._kh_gate_threshold)))

        self._kh_risk_history.append(x_t)

        # Fit Koopman from recent history
        hist = self._kh_risk_history
        W = self._kh_window
        L = self._kh_delay_L
        if len(hist) >= L + 3:
            seq = hist[-W - L:] if len(hist) > W + L else hist
            T_seq = len(seq)
            if T_seq >= L + 2:
                X_rows, Y_rows = [], []
                for t in range(L, T_seq - 1):
                    emb = np.concatenate([seq[t - i] for i in range(L)])
                    X_rows.append(emb)
                    Y_rows.append(seq[t + 1])
                X = np.array(X_rows)
                Y = np.array(Y_rows)
                m = X.shape[1]
                XtX = X.T @ X + self._kh_ridge * np.eye(m)
                XtY = X.T @ Y
                try:
                    self._kh_K_matrix = np.linalg.solve(XtX, XtY)
                except np.linalg.LinAlgError:
                    pass

        return gate_val

    def compute_snr_gate(self, grad_vec: torch.Tensor) -> float:
        """Compute gate value based on gradient consistency with EMA."""
        if not self._snr_gate_enabled:
            return 1.0

        if self._snr_gate_mu is None:
            self._snr_gate_mu = grad_vec.detach().clone()
            return 1.0

        cos_t = F.cosine_similarity(
            grad_vec.unsqueeze(0), self._snr_gate_mu.unsqueeze(0)).item()

        self._snr_gate_mu = (self._snr_gate_mu_beta * self._snr_gate_mu
                             + (1 - self._snr_gate_mu_beta) * grad_vec.detach())

        x = self._snr_gate_steepness * (cos_t - self._snr_gate_tau_cos)
        gate = 1.0 / (1.0 + math.exp(-x))
        return gate

    def _filter_gradient_c2(self, grad_vec: torch.Tensor,
                            loss_t: float = None, rho_t: float = None,
                            drift_t: float = None) -> torch.Tensor:
        """C2: Gradient subspace filtering (prompt gradients only)."""
        if self.c2_mode == "off":
            return grad_vec

        if self.c2_mode == "conflict_spectral":
            return self._filter_gradient_conflict_spectral(grad_vec)

        if self.c2_mode == "conflict_router":
            return self._filter_gradient_conflict_router(grad_vec)

        if self.c2_mode == "qp_decomp":
            return self._filter_gradient_qp_decomp(grad_vec)

        if self.c2_mode == "a_soft":
            return self._filter_gradient_a_soft(grad_vec)

        if self.c2_mode == "spectral_bridge":
            return self._filter_gradient_spectral_bridge(grad_vec)

        if self.c2_mode == "diag_shrink":
            return self._filter_gradient_diag_shrink(grad_vec)

        if self.c2_mode == "hankel_fa":
            return self._filter_gradient_hankel_fa(grad_vec)

        if self.c2_mode == "force_adaptive":
            return self._filter_gradient_force_adaptive(grad_vec)

        if self.c2_mode == "koopman_bal":
            return self._filter_gradient_koopman_bal(grad_vec)

        if self.c2_mode == "hbuo_peft" and self.peft_hbuo is not None:
            return self.peft_hbuo.filter_gradient(
                grad_vec, energy_threshold=self.c2_energy_threshold,
                loss_t=loss_t, rho_t=rho_t, drift_t=drift_t)

        g_fast = self.hbuo.filter_gradient_fast(
            grad_vec, energy_threshold=self.c2_energy_threshold)

        if self.c2_mode == "full":
            if (self.state_manager.projection_is_dynamic
                    and self.hbuo.has_hsv()):
                g_z = self.state_manager.projection.T @ grad_vec
                g_z_filtered = self.hbuo.filter_gradient(
                    g_z, energy_threshold=self.c2_energy_threshold)
                return self.state_manager.projection @ g_z_filtered

        return g_fast

    def _koopman_transport_gradient(
        self, grad_vec: torch.Tensor, A_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate gradient direction using the Koopman operator.

        Decomposes the gradient into a component in the Koopman state
        subspace and an orthogonal residual. Applies A^T to rotate the
        projected component, preserving the residual unchanged.

        g' = (I - U U^T) g + U A^T U^T g
        """
        U = self.state_manager.projection  # (d, r)
        g_z = U.T @ grad_vec               # (r,) — gradient in state space
        g_z_rotated = A_hat.T @ g_z        # (r,) — transported gradient
        g_proj = U @ g_z                    # (d,) — original projected
        g_proj_rotated = U @ g_z_rotated   # (d,) — rotated projected
        return grad_vec - g_proj + g_proj_rotated

    def _select_confident_samples(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """Select confident samples by entropy."""
        if self.single_view or logits.shape[0] <= 1:
            return logits

        batch_entropy = -(
            logits.softmax(dim=-1) * logits.log_softmax(dim=-1)
        ).sum(-1)
        idx = torch.argsort(batch_entropy)
        n_keep = max(
            min(4, logits.shape[0]),
            int(logits.shape[0] * self.selection_p))
        return logits[idx[:n_keep]]

    # ====================================================================
    # Koopman-SAM (T4)
    # ====================================================================

    def _sam_get_perturbation_direction(
        self, grad_vec: torch.Tensor
    ) -> torch.Tensor:
        """Compute SAM perturbation direction.

        Uses Koopman unstable modes if available, otherwise falls back
        to standard gradient-ascent direction.
        """
        traj = self.state_manager.get_trajectory_adaptive(
            max_window=self.koopman.W, min_window=2)
        if traj is None:
            g_norm = grad_vec.norm()
            return grad_vec / g_norm.clamp(min=1e-12)

        Z_0, Z_1 = traj
        K = self.koopman.fit_koopman(Z_0, Z_1)
        try:
            eigenvalues, eigenvectors = torch.linalg.eig(K.cpu())
        except (RuntimeError, ValueError):
            g_norm = grad_vec.norm()
            return grad_vec / g_norm.clamp(min=1e-12)

        magnitudes = eigenvalues.abs()
        unstable_mask = magnitudes > 1.0

        if unstable_mask.sum() == 0:
            g_norm = grad_vec.norm()
            return grad_vec / g_norm.clamp(min=1e-12)

        V_unstable = eigenvectors[:, unstable_mask].real.to(self.device)
        U = self.state_manager.projection

        g_z = self.state_manager.projection.T @ grad_vec
        proj = V_unstable @ (V_unstable.T @ g_z)
        direction = U @ proj

        d_norm = direction.norm()
        if d_norm < 1e-12:
            g_norm = grad_vec.norm()
            return grad_vec / g_norm.clamp(min=1e-12)
        return direction / d_norm

    def _sam_step(
        self,
        x: torch.Tensor,
        prompt_params: List[nn.Parameter],
        head_params: List[nn.Parameter],
        first_prompt_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """Koopman-SAM: compute sharpness-aware gradient.

        1. Compute perturbation direction (Koopman unstable modes or gradient)
        2. Perturb prompt → forward → gradient at perturbed point
        3. Restore prompt → return perturbed-point gradient

        Returns:
            (sam_prompt_grad, sam_head_grads)
        """
        direction = self._sam_get_perturbation_direction(first_prompt_grad)

        originals = [p.data.clone() for p in prompt_params]
        offset = 0
        for p in prompt_params:
            numel = p.numel()
            p.data.add_(
                self.sam_rho * direction[offset:offset + numel].reshape(p.shape))
            offset += numel

        loss_p, _ = self._compute_loss(x)

        all_params = prompt_params + head_params
        grads_p = torch.autograd.grad(
            loss_p, all_params, create_graph=False, allow_unused=True)

        for p, orig in zip(prompt_params, originals):
            p.data.copy_(orig)

        n_prompt = len(prompt_params)
        sam_prompt_grad = torch.cat([
            (g.detach().reshape(-1) if g is not None
             else torch.zeros(p.numel(), device=self.device))
            for g, p in zip(grads_p[:n_prompt], prompt_params)
        ])
        sam_head_grads = [
            g.detach() if g is not None else None
            for g in grads_p[n_prompt:]
        ]
        return sam_prompt_grad, sam_head_grads

    # ====================================================================
    # KTMV
    # ====================================================================

    def _ktmv_is_ready(self) -> bool:
        if not self.ktmv_enabled or self.ktmv is None:
            return False
        return len(self.state_manager._buffer) > self.koopman.W

    def _ktmv_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass producing KTMV virtual views."""
        prompt_params = self.state_manager.get_prompt_params(self.model)

        self.model.train()
        logits_0 = self.model(x)
        views = [logits_0]

        traj = self.state_manager.get_trajectory_adaptive(
            max_window=self.koopman.W, min_window=2)
        if traj is None:
            return views

        Z_0, Z_1 = traj
        K = self.koopman.fit_koopman(Z_0, Z_1)
        z_t = self.state_manager.prompt_to_state(self.model)

        perturbations = self.ktmv.generate_prompt_perturbations(
            K, z_t, self.state_manager.projection)

        if not perturbations:
            return views

        originals = [p.data.clone() for p in prompt_params]

        for delta_p in perturbations:
            offset = 0
            for p, orig in zip(prompt_params, originals):
                numel = p.numel()
                p.data.copy_(
                    orig + delta_p[offset:offset + numel].reshape(p.shape))
                offset += numel

            with torch.no_grad():
                self.model.eval()
                logits_m = self.model(x)
            views.append(logits_m)

        for p, orig in zip(prompt_params, originals):
            p.data.copy_(orig)
        self.model.train()

        return views

    # ====================================================================
    # Model forward helpers
    # ====================================================================

    def _forward_with_features(self, x: torch.Tensor):
        """Forward returning (features, logits). Falls back gracefully.

        Handles ViT wrapper (model.enc + model.head) where enc.head may
        be Identity.  Always routes CLS features through the outer head
        to produce correctly-shaped logits.
        """
        self.model.eval()
        enc = getattr(self.model, "enc", self.model)
        outer_head = self.model.head if hasattr(self.model, "enc") else None
        if hasattr(enc, "forward_features"):
            cls_feat, raw_logits, _ = enc.forward_features(x)
            logits = outer_head(cls_feat) if outer_head is not None else raw_logits
            return cls_feat, logits
        logits = self.model(x)
        return None, logits

    def _forward_with_attn(self, x: torch.Tensor):
        """Forward returning (logits, attn_weights).

        Used by _compute_loss when attention entropy loss is active.
        """
        enc = getattr(self.model, "enc", self.model)
        outer_head = self.model.head if hasattr(self.model, "enc") else None
        if hasattr(enc, "forward_features"):
            cls_feat, raw_logits, attn_weights = enc.forward_features(x)
            logits = outer_head(cls_feat) if outer_head is not None else raw_logits
            return logits, attn_weights
        logits = self.model(x)
        return logits, None

    @torch.no_grad()
    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(x)

    # ====================================================================
    # Step 2 variants: original vs gradient-reuse
    # ====================================================================

    def _step2_original(
        self, x_for_adapt, prompt_params, head_params,
        n_prompt, use_optim, info,
    ):
        """Original K loop: full forward+backward each step."""
        all_params = prompt_params + list(head_params)

        for k in range(self.steps_per_sample):
            if use_optim and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, _ = self._compute_loss(x_for_adapt)
            else:
                loss, _ = self._compute_loss(x_for_adapt)
            info["entropy"] = loss.item()

            if not torch.isfinite(loss):
                break

            scaler_active = False
            if use_optim:
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    scaler_active = True
                else:
                    loss.backward()

                prompt_grad_vec = torch.cat([
                    (p.grad.detach().reshape(-1) if p.grad is not None
                     else torch.zeros(p.numel(), device=self.device))
                    for p in prompt_params
                ])
                head_grads = [p.grad for p in head_params] if head_params else []
            else:
                grads = torch.autograd.grad(
                    loss, all_params, create_graph=False, allow_unused=True)
                prompt_grad_vec = torch.cat([
                    (g.detach().reshape(-1) if g is not None
                     else torch.zeros(p.numel(), device=self.device))
                    for g, p in zip(grads[:n_prompt], prompt_params)
                ])
                head_grads = list(grads[n_prompt:])

            skip_step = False

            if not torch.isfinite(prompt_grad_vec).all():
                skip_step = True

            if not skip_step and self.sam_enabled:
                prompt_grad_vec, head_grads = self._sam_step(
                    x_for_adapt, prompt_params, head_params,
                    prompt_grad_vec)
                if not torch.isfinite(prompt_grad_vec).all():
                    skip_step = True

            if not skip_step:
                z_t = self.state_manager.prompt_to_state(self.model)
                self.state_manager.update_buffer(z_t)

                traj = self.state_manager.get_trajectory_adaptive(
                    max_window=self.koopman.W, min_window=2)
                if traj is not None:
                    Z_0, Z_1 = traj
                    eta_t, koopman_info = self.koopman.get_adapted_lr(
                        Z_0, Z_1)
                else:
                    eta_t, koopman_info = self.koopman.get_adapted_lr(
                        None, None)

                info["rho"] = koopman_info["rho"]
                info["eta"] = eta_t

                if koopman_info["rollback"]:
                    self._rollback_to_stable()
                    skip_step = True

                if koopman_info["early_stop"]:
                    skip_step = True

            if not skip_step:
                if self.collect_diagnostics:
                    self.diagnostics_log["grad_norm_raw"].append(
                        prompt_grad_vec.norm().item())
                _c2_loss = info.get("entropy", None)
                _c2_rho = info.get("rho", None)
                _c2_drift = info.get("prompt_drift", None)
                prompt_grad_vec = self._filter_gradient_c2(
                    prompt_grad_vec, loss_t=_c2_loss,
                    rho_t=_c2_rho, drift_t=_c2_drift)
                if self.collect_diagnostics:
                    self.diagnostics_log["grad_norm_filtered"].append(
                        prompt_grad_vec.norm().item())

                # Koopman-Hankel risk gate
                if k == 0:
                    _ent = info.get("entropy", 0.0)
                    _conf_est = max(0.0, 1.0 - _ent / 5.3)
                    kh_gate = self.compute_kh_gate(
                        _ent, _conf_est,
                        info.get("rho", 0.0), prompt_grad_vec)
                    info["kh_gate"] = kh_gate
                else:
                    kh_gate = info.get("kh_gate", 1.0)
                eta_t = eta_t * kh_gate

            if skip_step:
                if scaler_active:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                break

            # Anchor regularization: add 2λ(P - P₀) to gradient
            if self.anchor_lambda > 0 and self._anchor_prompt_state is not None:
                anchor_parts = []
                for i, p in enumerate(prompt_params):
                    p0 = self._anchor_prompt_state[i]
                    anchor_parts.append(
                        2.0 * self.anchor_lambda * (p.data - p0).reshape(-1))
                anchor_vec = torch.cat(anchor_parts).to(prompt_grad_vec.dtype)
                prompt_grad_vec = prompt_grad_vec + anchor_vec

            if use_optim:
                lr_scale = eta_t / self.base_lr
                offset = 0
                for p in prompt_params:
                    numel = p.numel()
                    filtered = prompt_grad_vec[offset:offset + numel].reshape(
                        p.shape)
                    p.grad.copy_(filtered * lr_scale)
                    offset += numel

                if head_params:
                    for p in head_params:
                        if p.grad is not None:
                            p.grad.clamp_(
                                -self.head_grad_clip, self.head_grad_clip)
                            p.grad.mul_(self.head_lr_mult * lr_scale)

                if scaler_active:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
            else:
                offset = 0
                for p in prompt_params:
                    numel = p.numel()
                    update = prompt_grad_vec[offset:offset + numel].reshape(
                        p.shape)
                    p.data.sub_(eta_t * update)
                    offset += numel

                if head_params:
                    for p, g in zip(head_params, head_grads):
                        if g is not None:
                            g_clip = g.detach().clamp(
                                -self.head_grad_clip, self.head_grad_clip)
                            p.data.sub_(eta_t * self.head_lr_mult * g_clip)

            if self.ema_enabled:
                self._update_ema()

    def _step2_grad_reuse(
        self, x_for_adapt, prompt_params, head_params,
        n_prompt, use_optim, info,
    ):
        """Gradient-reuse K loop: 1 forward+backward, K cheap C1/C2/update.

        Caches image features and computes gradient once. Subsequent K-1
        steps reuse the gradient direction, relying on C1 (spectral lr)
        and C2 (subspace filtering) for per-step adaptation.
        """
        # --- Phase 1: Cache image features + single forward+backward ---
        cached_img_feat = None
        has_cache = hasattr(self.model, 'encode_image_cached')
        if has_cache:
            cached_img_feat = self.model.encode_image_cached(x_for_adapt)

        if has_cache and cached_img_feat is not None:
            if use_optim and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, _ = self._compute_loss_cached(cached_img_feat)
            else:
                loss, _ = self._compute_loss_cached(cached_img_feat)
        else:
            if use_optim and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, _ = self._compute_loss(x_for_adapt)
            else:
                loss, _ = self._compute_loss(x_for_adapt)
        info["entropy"] = loss.item()

        # Estimate confidence for KH risk gate
        with torch.no_grad():
            if has_cache and cached_img_feat is not None:
                _kh_logits = self.model.forward_from_cache(cached_img_feat)
            else:
                _kh_logits = self.model(x_for_adapt)
            _kh_probs = F.softmax(_kh_logits, dim=1)
            info["confidence"] = _kh_probs.max(dim=1).values.mean().item()

        if not torch.isfinite(loss):
            return

        scaler_active = False
        if use_optim:
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                scaler_active = True
            else:
                loss.backward()

            base_prompt_grad = torch.cat([
                (p.grad.detach().reshape(-1) if p.grad is not None
                 else torch.zeros(p.numel(), device=self.device))
                for p in prompt_params
            ]).clone()
            base_head_grads = (
                [p.grad.detach().clone() for p in head_params]
                if head_params else []
            )
        else:
            all_params = prompt_params + list(head_params)
            grads = torch.autograd.grad(
                loss, all_params, create_graph=False, allow_unused=True)
            base_prompt_grad = torch.cat([
                (g.detach().reshape(-1) if g is not None
                 else torch.zeros(p.numel(), device=self.device))
                for g, p in zip(grads[:n_prompt], prompt_params)
            ]).clone()
            base_head_grads = [
                g.detach().clone() if g is not None else None
                for g in grads[n_prompt:]
            ]

        if not torch.isfinite(base_prompt_grad).all():
            if scaler_active:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            return

        if self.sam_enabled:
            base_prompt_grad, base_head_grads = self._sam_step(
                x_for_adapt, prompt_params, head_params,
                base_prompt_grad)
            if not torch.isfinite(base_prompt_grad).all():
                if scaler_active:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                return

        # --- Phase 2: K iterations with gradient reuse + C1/C2 per step ---
        # Koopman gradient transport: rotate gradient in state subspace
        # using the fitted Koopman operator A each step.
        _A_hat = None
        current_grad = base_prompt_grad.clone()

        for k in range(self.steps_per_sample):
            skip_step = False

            # C1: Koopman spectral control
            z_t = self.state_manager.prompt_to_state(self.model)
            self.state_manager.update_buffer(z_t)

            traj = self.state_manager.get_trajectory_adaptive(
                max_window=self.koopman.W, min_window=2)
            if traj is not None:
                Z_0, Z_1 = traj
                eta_t, koopman_info = self.koopman.get_adapted_lr(Z_0, Z_1)
                if self.koopman_grad_transport:
                    _A_hat = self.koopman.fit_koopman(Z_0, Z_1)
            else:
                eta_t, koopman_info = self.koopman.get_adapted_lr(None, None)

            # Koopman gradient transport for k > 0
            if (k > 0 and self.koopman_grad_transport
                    and _A_hat is not None):
                current_grad = self._koopman_transport_gradient(
                    current_grad, _A_hat)
            prompt_grad_vec = current_grad.clone()

            info["rho"] = koopman_info["rho"]
            info["eta"] = eta_t

            if koopman_info["rollback"]:
                self._rollback_to_stable()
                skip_step = True
            if koopman_info["early_stop"]:
                skip_step = True

            # C2: Gradient subspace filtering (applied fresh each step)
            if not skip_step:
                _c2_loss = info.get("entropy", None)
                _c2_rho = info.get("rho", None)
                _c2_drift = info.get("prompt_drift", None)
                prompt_grad_vec = self._filter_gradient_c2(
                    prompt_grad_vec, loss_t=_c2_loss,
                    rho_t=_c2_rho, drift_t=_c2_drift)

                # SNR/consensus gate: scale eta_t based on gradient quality
                snr_gate = self.compute_snr_gate(prompt_grad_vec)
                eta_t = eta_t * snr_gate

                # Koopman-Hankel risk gate (only on first step to avoid recomputing)
                if k == 0:
                    _loss_for_kh = info.get("entropy", 0.0)
                    _conf_for_kh = info.get("confidence", 0.5)
                    _rho_for_kh = info.get("rho", 0.0)
                    kh_gate = self.compute_kh_gate(
                        _loss_for_kh, _conf_for_kh, _rho_for_kh,
                        prompt_grad_vec)
                    info["kh_gate"] = kh_gate
                else:
                    kh_gate = info.get("kh_gate", 1.0)
                eta_t = eta_t * kh_gate

            if skip_step:
                if k == 0 and scaler_active:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                break

            # Apply update (manual step — bypass optimizer for k>0 to
            # avoid stale momentum state from the single backward pass)
            if k == 0 and use_optim:
                lr_scale = eta_t / self.base_lr
                offset = 0
                for p in prompt_params:
                    numel = p.numel()
                    filtered = prompt_grad_vec[offset:offset + numel].reshape(
                        p.shape)
                    p.grad.copy_(filtered * lr_scale)
                    offset += numel

                if head_params:
                    for i, p in enumerate(head_params):
                        if p.grad is not None:
                            p.grad.copy_(base_head_grads[i] * lr_scale
                                         * self.head_lr_mult)
                            p.grad.clamp_(
                                -self.head_grad_clip, self.head_grad_clip)

                if scaler_active:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
            else:
                offset = 0
                for p in prompt_params:
                    numel = p.numel()
                    update = prompt_grad_vec[offset:offset + numel].reshape(
                        p.shape)
                    p.data.sub_(eta_t * update)
                    offset += numel

                if head_params and base_head_grads:
                    for p, g in zip(head_params, base_head_grads):
                        if g is not None:
                            g_clip = g.clamp(
                                -self.head_grad_clip, self.head_grad_clip)
                            p.data.sub_(eta_t * self.head_lr_mult * g_clip)

            if self.ema_enabled:
                self._update_ema()

    def _step2_view_subbatch(
        self, x_for_adapt, prompt_params, head_params,
        n_prompt, use_optim, info,
    ):
        """VSGA view sub-batch: parallel gradient collection + SVD denoising.

        1. Cache image features once (frozen encoder).
        2. Split views into G groups, compute per-group gradient independently
           WITHOUT updating the prompt between groups.
        3. Pass G gradient vectors to HBUO.filter_gradient_vsga() for SVD-based
           denoising — keeps only directions that are consistent across groups.
        4. Use GAR (Gradient Agreement Ratio) from SVD to control step size
           via KoopmanController.get_lr_from_gar().
        5. Apply a single denoised update.
        """
        has_cache = hasattr(self.model, 'encode_image_cached')
        if not has_cache:
            self._step2_original(
                x_for_adapt, prompt_params, head_params,
                n_prompt, use_optim, info)
            return

        n_views = x_for_adapt.shape[0]
        G = min(self.n_view_groups, n_views)
        if G < 2:
            self._step2_original(
                x_for_adapt, prompt_params, head_params,
                n_prompt, use_optim, info)
            return

        cached_img_feat = self.model.encode_image_cached(x_for_adapt)

        group_size = n_views // G
        remainder = n_views % G
        groups = []
        start = 0
        for g in range(G):
            end = start + group_size + (1 if g < remainder else 0)
            groups.append((start, end))
            start = end

        # --- Phase 1: Collect per-group gradients (no prompt update) ---
        group_grads = []
        last_entropy = 0.0

        for g_idx, (g_start, g_end) in enumerate(groups):
            group_feat = cached_img_feat[g_start:g_end]

            self.model.train()
            logits = self.model.forward_from_cache(group_feat)
            logits_conf = self._select_confident_samples(logits)
            loss = self._entropy_loss(logits_conf)

            last_entropy = loss.item()
            if not torch.isfinite(loss):
                continue

            all_params = prompt_params + list(head_params)
            grads = torch.autograd.grad(
                loss, all_params, create_graph=False, allow_unused=True)

            prompt_grad_vec = torch.cat([
                (g.detach().reshape(-1) if g is not None
                 else torch.zeros(p.numel(), device=self.device))
                for g, p in zip(grads[:n_prompt], prompt_params)
            ])

            if torch.isfinite(prompt_grad_vec).all():
                group_grads.append(prompt_grad_vec)

        info["entropy"] = last_entropy

        if len(group_grads) == 0:
            return

        # --- Phase 1b (fullgrad mode): compute full-batch gradient ---
        fullgrad_vec = None
        if self.vsga_mode == "fullgrad":
            self.model.train()
            logits_full = self.model.forward_from_cache(cached_img_feat)
            logits_full_conf = self._select_confident_samples(logits_full)
            loss_full = self._entropy_loss(logits_full_conf)
            info["entropy"] = loss_full.item()
            if torch.isfinite(loss_full):
                all_params = prompt_params + list(head_params)
                grads_full = torch.autograd.grad(
                    loss_full, all_params,
                    create_graph=False, allow_unused=True)
                fullgrad_vec = torch.cat([
                    (g.detach().reshape(-1) if g is not None
                     else torch.zeros(p.numel(), device=self.device))
                    for g, p in zip(grads_full[:n_prompt], prompt_params)
                ])
                if not torch.isfinite(fullgrad_vec).all():
                    fullgrad_vec = None

        # --- Phase 2: GAR computation + optional VSGA denoising ---
        g_filtered, gar, vsga_info = self.hbuo.filter_gradient_vsga(
            group_grads, energy_threshold=self.c2_energy_threshold)

        if self.vsga_mode == "fullgrad" and fullgrad_vec is not None:
            g_filtered = fullgrad_vec

        if self.collect_diagnostics:
            self.diagnostics_log["grad_norm_raw"].append(
                torch.stack(group_grads).mean(dim=0).norm().item())
            self.diagnostics_log["grad_norm_filtered"].append(
                g_filtered.norm().item())
            if vsga_info.get("svd_spectrum"):
                self.diagnostics_log["grad_sv_raw"].append(
                    vsga_info["svd_spectrum"])
            self.diagnostics_log["c2_effective_rank"].append(
                vsga_info.get("k_eff", 0))

        # --- Phase 3: GAR-based step-size control (C1) ---
        eta_t, koopman_info = self.koopman.get_lr_from_gar(gar)

        info["rho"] = koopman_info.get("gar", gar)
        info["eta"] = eta_t
        info["gar"] = gar
        info["k_eff"] = vsga_info.get("k_eff", 0)

        if koopman_info.get("rollback", False):
            self._rollback_to_stable()
            return

        if koopman_info.get("early_stop", False):
            return

        # --- Phase 3.5: Online drift protection ---
        if self.max_prompt_drift > 0 and self.protocol == "online":
            drift = self.state_manager.prompt_drift(self.model)
            if drift > self.max_prompt_drift:
                drift_scale = max(0.01, self.max_prompt_drift / drift)
                eta_t *= drift_scale
                info["eta"] = eta_t
                info["drift_scale"] = drift_scale

        # --- Phase 3.7: Anchor regularization ∇(λ||P-P₀||²) = 2λ(P-P₀) ---
        if self.anchor_lambda > 0 and self._anchor_prompt_state is not None:
            anchor_grads = []
            for i, p in enumerate(prompt_params):
                p0 = self._anchor_prompt_state[i]
                anchor_grads.append(
                    2.0 * self.anchor_lambda * (p.data - p0).reshape(-1))
            anchor_grad_vec = torch.cat(anchor_grads).to(g_filtered.dtype)
            g_filtered = g_filtered + anchor_grad_vec

        # --- Phase 4: Single denoised update ---
        if use_optim:
            self.optimizer.zero_grad()
            lr_scale = eta_t / self.base_lr
            offset = 0
            for p in prompt_params:
                numel = p.numel()
                synthetic_grad = g_filtered[offset:offset + numel].reshape(
                    p.shape)
                p.grad = (synthetic_grad * lr_scale).to(p.dtype)
                offset += numel

            self.optimizer.step()
        else:
            offset = 0
            for p in prompt_params:
                numel = p.numel()
                update = g_filtered[offset:offset + numel].reshape(p.shape)
                p.data.sub_(eta_t * update)
                offset += numel

        if self.ema_enabled:
            self._update_ema()

    # ====================================================================
    # Main entry point
    # ====================================================================

    def adapt_and_predict(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
        x_adapt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Perform TTA on a single batch and return predictions."""
        x = x.to(self.device)
        x_for_adapt = x_adapt.to(self.device) if x_adapt is not None else x
        self._sample_count += 1

        info = {
            "rho": 0.0, "eta": self.base_lr,
            "entropy": 0.0, "drift": 0.0,
        }

        # --- Step 0: Episodic reset ---
        if self.protocol == "episodic":
            self._reset_for_episodic()

        # --- Step 0.5: Confidence gating + initial prediction ---
        _cached_x_img_feat = None
        has_cache = ((self.grad_reuse or self.view_subbatch)
                     and hasattr(self.model, 'encode_image_cached'))
        with torch.no_grad():
            self.model.eval()
            if has_cache and not self.proto_enabled:
                _cached_x_img_feat = self.model.encode_image_cached(x)
                init_logits = self.model.forward_from_cache(
                    _cached_x_img_feat)
                init_features = None
            elif self.proto_enabled:
                init_features, init_logits = self._forward_with_features(x)
            else:
                init_logits = self.model(x)
                init_features = None
            init_entropy = self._entropy_loss(init_logits).item()
            info["entropy"] = init_entropy

        if (self.confidence_threshold > 0
                and init_entropy < self.confidence_threshold):
            predictions = init_logits.argmax(dim=-1)
            info["skipped"] = True
            info["drift"] = self.state_manager.prompt_drift(self.model)
            if y is not None:
                y = y.to(self.device)
                acc = (predictions == y).float().mean().item()
                self.metrics["accuracy"].append(acc)
                info["accuracy"] = acc
            return predictions, info

        # --- Step 1: C2-full projection + Gramian ---
        if self.c2_mode == "full":
            rebuilt = self.hbuo.maybe_rebuild_projection(self.state_manager)

            if self.state_manager.projection_is_dynamic:
                self._sample_count_gramian = getattr(
                    self, '_sample_count_gramian', 0) + 1
                need_gramian = (
                    rebuilt
                    or not self.hbuo.has_hsv()
                    or self._sample_count_gramian % self.gramian_update_freq == 0
                )
                if need_gramian:
                    was_training = self.model.training
                    self.model.eval()
                    try:
                        self.hbuo.update_gramians(
                            self.model, x, self.state_manager)
                    except (RuntimeError, ValueError):
                        pass
                    if was_training:
                        self.model.train()

        # --- Step 2: K inner update steps ---
        prompt_params, head_params = self._get_all_trainable_params()
        all_params = prompt_params + head_params
        n_prompt = len(prompt_params)
        use_optim = self.optimizer is not None

        if self.view_subbatch and x_for_adapt.shape[0] > 1:
            self._step2_view_subbatch(
                x_for_adapt, prompt_params, head_params,
                n_prompt, use_optim, info)
        elif self.grad_reuse and self.steps_per_sample > 1:
            self._step2_grad_reuse(
                x_for_adapt, prompt_params, head_params,
                n_prompt, use_optim, info)
        else:
            self._step2_original(
                x_for_adapt, prompt_params, head_params,
                n_prompt, use_optim, info)

        # --- Step 3: Save stable state ---
        if info["rho"] < 1.0 or info["rho"] == 0.0:
            self._save_stable_state()

        # --- Step 4: Final prediction ---
        with torch.no_grad():
            self.model.eval()
            if self.proto_enabled:
                final_features, final_logits = self._forward_with_features(x)
            elif _cached_x_img_feat is not None:
                final_logits = self.model.forward_from_cache(
                    _cached_x_img_feat)
                final_features = None
            else:
                final_logits = self.model(x)
                final_features = None

            if self.ensemble_alpha > 0:
                final_logits = (self.ensemble_alpha * init_logits
                                + (1 - self.ensemble_alpha) * final_logits)

            # KDMD
            if self.kdmd_enabled:
                per_preds = final_logits.argmax(dim=-1)
                for i in range(final_logits.shape[0]):
                    phi_b = self.kernel_lifter.lift(init_logits[i])
                    phi_a = self.kernel_lifter.lift(final_logits[i])
                    self.kdmd.acc.update(
                        per_preds[i].item(), phi_b, phi_a)

                if self.kdmd.is_ready():
                    for i in range(final_logits.shape[0]):
                        phi_b = self.kernel_lifter.lift(init_logits[i])
                        phi_a = self.kernel_lifter.lift(final_logits[i])
                        delta = self.kdmd.compute_alignment(phi_b, phi_a)
                        final_logits[i] = final_logits[i] + delta

            # T2: Prototype rectification
            if (self.proto_enabled
                    and self.proto_bank is not None
                    and final_features is not None):
                self.proto_bank.update(final_features, final_logits)
                final_logits = self.proto_bank.rectify_logits(
                    final_features, final_logits)

            predictions = final_logits.argmax(dim=-1)

        # --- Metrics ---
        info["drift"] = self.state_manager.prompt_drift(self.model)
        self.metrics["entropy"].append(info["entropy"])
        self.metrics["rho"].append(info["rho"])
        self.metrics["eta"].append(info["eta"])
        self.metrics["drift"].append(info["drift"])

        # --- Diagnostics ---
        if self.collect_diagnostics:
            self.diagnostics_log["rho_per_sample"].append(info["rho"])
            self.diagnostics_log["prediction_before"].append(
                init_logits.argmax(dim=-1).cpu().tolist())
            self.diagnostics_log["prediction_after"].append(
                final_logits.argmax(dim=-1).cpu().tolist())
            if self.hbuo.last_svd_spectrum is not None:
                self.diagnostics_log["grad_sv_raw"].append(
                    self.hbuo.last_svd_spectrum.tolist())
                self.diagnostics_log["c2_effective_rank"].append(
                    self.hbuo.last_k_eff)

        if y is not None:
            y = y.to(self.device)
            acc = (predictions == y).float().mean().item()
            self.metrics["accuracy"].append(acc)
            info["accuracy"] = acc

        return predictions, info

    def get_summary(self) -> Dict:
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_last"] = values[-1]
        summary["total_samples"] = self._sample_count
        return summary
