"""
KDMD: Kernel Dynamic Mode Decomposition for dynamics-derived semantic anchors.

Provides class-specific Koopman operators learned online from prompt adaptation
dynamics.  During TTA each sample's state transition (z_before -> z_after) is
kernel-lifted and accumulated per predicted class.  The resulting per-class
operators serve as *semantic anchors*: for a new sample the alignment between
its observed dynamics and each class operator yields a logit correction that
steers prediction toward dynamically consistent classes.

Three components:
  KernelLifter           — Random Fourier Feature (RFF) kernel lifting
  ClassKoopmanAccumulator — per-class running EDMD in lifted space
  KDMDPredictor          — alignment scoring + logit enhancement
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class KernelLifter:
    """Random Fourier Features (RFF) approximation of the RBF kernel.

    Maps z in R^r  ->  phi(z) in R^D   where
        phi(z) = sqrt(2/D) * cos(z @ W + b)
    and the inner product <phi(z1), phi(z2)> approximates
        k(z1, z2) = exp(-gamma * ||z1-z2||^2).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        gamma: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.input_dim = input_dim
        self.D = output_dim
        self.gamma = gamma
        self.device = device or torch.device("cpu")

        self.W = (torch.randn(input_dim, output_dim, device=self.device)
                  * math.sqrt(2.0 * gamma))
        self.b = (torch.rand(output_dim, device=self.device) * 2.0 * math.pi)
        self._scale = math.sqrt(2.0 / output_dim)

    @torch.no_grad()
    def lift(self, z: torch.Tensor) -> torch.Tensor:
        """Lift state vector(s) to RFF space.

        Args:
            z: (..., r) state vector or batch of vectors.
        Returns:
            phi: (..., D) lifted vector(s).
        """
        return self._scale * torch.cos(z @ self.W + self.b)

    def to(self, device: torch.device) -> "KernelLifter":
        self.W = self.W.to(device)
        self.b = self.b.to(device)
        self.device = device
        return self


class ClassKoopmanAccumulator:
    """Maintains per-class Koopman operators via running EDMD.

    For each class c the running statistics are:
        gram_c  = sum_i phi(z_i^before) phi(z_i^before)^T   (D x D)
        cross_c = sum_i phi(z_i^before) phi(z_i^after)^T    (D x D)
    so the EDMD operator is  K_c = (gram_c + lambda*I)^{-1} cross_c.

    Using rank-1 running updates keeps memory at O(C * D^2) and cost at
    O(D^2) per sample, independent of stream length.
    """

    def __init__(
        self,
        num_classes: int,
        lifted_dim: int,
        ridge_lambda: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        self.C = num_classes
        self.D = lifted_dim
        self.ridge = ridge_lambda
        self.device = device or torch.device("cpu")

        self.gram = torch.zeros(num_classes, lifted_dim, lifted_dim,
                                device=self.device)
        self.cross = torch.zeros(num_classes, lifted_dim, lifted_dim,
                                 device=self.device)
        self.counts = torch.zeros(num_classes, device=self.device)

    @torch.no_grad()
    def update(self, class_idx: int, phi_before: torch.Tensor,
               phi_after: torch.Tensor):
        """Rank-1 update for class `class_idx`."""
        pb = phi_before.detach()
        pa = phi_after.detach()
        self.gram[class_idx] += pb.unsqueeze(1) * pb.unsqueeze(0)
        self.cross[class_idx] += pb.unsqueeze(1) * pa.unsqueeze(0)
        self.counts[class_idx] += 1

    @torch.no_grad()
    def get_operator(self, class_idx: int) -> Optional[torch.Tensor]:
        """Compute K_c = (gram_c + lambda*I)^{-1} cross_c  on CPU."""
        if self.counts[class_idx] < 3:
            return None
        G = self.gram[class_idx].cpu()
        X = self.cross[class_idx].cpu()
        reg = self.ridge * self.counts[class_idx].item() * torch.eye(self.D)
        try:
            K = torch.linalg.solve(G + reg, X)
        except (RuntimeError, ValueError):
            return None
        return K.to(self.device)

    def ready_classes(self, min_samples: int = 3) -> torch.Tensor:
        """Boolean mask of classes with enough samples."""
        return self.counts >= min_samples

    def to(self, device: torch.device) -> "ClassKoopmanAccumulator":
        self.gram = self.gram.to(device)
        self.cross = self.cross.to(device)
        self.counts = self.counts.to(device)
        self.device = device
        return self


class KDMDPredictor:
    """Dynamics-aligned prediction via per-class Koopman alignment.

    For a new sample with observed transition (phi_before, phi_after):
      score_c = ||K_c @ phi_before - phi_after||^2

    Lower score means the class-c dynamics better predict what actually
    happened — analogous to a sample being "closer" to class c's semantic
    anchor.

    The scores are converted to a logit adjustment:
      delta_logits = -softmax(scores / temperature)
    (negated because lower distance = better match → higher logit)
    """

    def __init__(
        self,
        accumulator: ClassKoopmanAccumulator,
        lam: float = 1.0,
        temperature: float = 1.0,
        min_class_coverage: float = 0.5,
        min_samples_per_class: int = 3,
    ):
        self.acc = accumulator
        self.lam = lam
        self.temperature = temperature
        self.min_coverage = min_class_coverage
        self.min_samples = min_samples_per_class
        self._cached_operators: Optional[torch.Tensor] = None
        self._cache_step = -1

    def is_ready(self) -> bool:
        """True when enough classes have sufficient samples."""
        ready = self.acc.ready_classes(self.min_samples)
        coverage = ready.float().mean().item()
        return coverage >= self.min_coverage

    @torch.no_grad()
    def _refresh_operators(self):
        """Batch-compute all class operators (cached, refreshed periodically)."""
        ops = []
        mask = []
        for c in range(self.acc.C):
            K = self.acc.get_operator(c)
            if K is not None:
                ops.append(K)
                mask.append(True)
            else:
                ops.append(torch.zeros(self.acc.D, self.acc.D,
                                       device=self.acc.device))
                mask.append(False)
        self._cached_operators = torch.stack(ops)          # (C, D, D)
        self._cached_mask = torch.tensor(mask, device=self.acc.device)

    @torch.no_grad()
    def compute_alignment(
        self,
        phi_before: torch.Tensor,
        phi_after: torch.Tensor,
        refresh_interval: int = 20,
    ) -> torch.Tensor:
        """Compute logit adjustment from dynamics alignment.

        Args:
            phi_before: (D,) lifted state before TTA.
            phi_after:  (D,) lifted state after TTA.
            refresh_interval: recompute operators every N calls.

        Returns:
            delta_logits: (C,) logit adjustment vector.
        """
        step = int(self.acc.counts.sum().item())
        if (self._cached_operators is None
                or step - self._cache_step >= refresh_interval):
            self._refresh_operators()
            self._cache_step = step

        K_all = self._cached_operators                      # (C, D, D)
        predicted = K_all @ phi_before                      # (C, D)
        diff = predicted - phi_after.unsqueeze(0)           # (C, D)
        scores = (diff * diff).sum(dim=-1)                  # (C,)

        # Mask out classes without enough data (give them neutral score)
        median_score = scores[self._cached_mask].median() if self._cached_mask.any() else 0.0
        scores[~self._cached_mask] = median_score

        neg_scores = -scores / (self.temperature + 1e-8)
        delta = F.log_softmax(neg_scores, dim=0)
        return self.lam * delta
