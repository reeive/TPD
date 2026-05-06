"""
Online Feature Prototypes with Dynamic Rectification (OFP-DR).

Non-parametric Bayesian posterior approximation: each class prototype
mu_c tracks the mean feature via EMA.  At prediction time, cosine
similarity between the test feature and all prototypes provides a
complementary classification signal to the parametric head.

Final prediction:
    logits_final = head_logits + gamma * cos_sim(f, mu_c) / tau
"""

import torch
import torch.nn.functional as F
from typing import Optional


class OnlinePrototypeBank:
    """Maintain per-class feature centroids from confident predictions.

    Attributes:
        num_classes: Number of classes.
        feat_dim: Feature dimensionality.
        gamma: Weight for prototype logits relative to head logits.
        tau: Temperature for cosine similarity.
        momentum: EMA momentum for prototype updates.
        conf_threshold: Minimum top-1 probability to update prototypes.
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        gamma: float = 1.0,
        tau: float = 0.1,
        momentum: float = 0.9,
        conf_threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.gamma = gamma
        self.tau = tau
        self.momentum = momentum
        self.conf_threshold = conf_threshold
        self.device = device or torch.device("cpu")

        self.prototypes = torch.zeros(
            num_classes, feat_dim, device=self.device)
        self.counts = torch.zeros(num_classes, device=self.device)
        self._min_coverage = 0.1

    @torch.no_grad()
    def update(self, features: torch.Tensor, logits: torch.Tensor):
        """Update prototypes with confident predictions.

        Args:
            features: CLS features, shape (B, D).
            logits: Head logits, shape (B, C).
        """
        probs = F.softmax(logits, dim=-1)
        max_prob, pred_class = probs.max(dim=-1)

        for i in range(features.shape[0]):
            if max_prob[i] < self.conf_threshold:
                continue
            c = pred_class[i].item()
            f = features[i]
            f = F.normalize(f, dim=0)

            if self.counts[c] == 0:
                self.prototypes[c] = f
            else:
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c]
                    + (1 - self.momentum) * f
                )
                self.prototypes[c] = F.normalize(
                    self.prototypes[c], dim=0)
            self.counts[c] += 1

    def is_ready(self) -> bool:
        """Check if enough classes have prototypes for meaningful rectification."""
        coverage = (self.counts > 0).float().mean().item()
        return coverage >= self._min_coverage

    @torch.no_grad()
    def rectify_logits(
        self,
        features: torch.Tensor,
        head_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Combine head logits with prototype-based cosine similarity.

        Args:
            features: CLS features, shape (B, D).
            head_logits: Head logits, shape (B, C).

        Returns:
            Rectified logits, shape (B, C).
        """
        if not self.is_ready():
            return head_logits

        f_norm = F.normalize(features, dim=-1)
        active_mask = (self.counts > 0).float()

        sim = f_norm @ self.prototypes.T
        sim = sim / self.tau
        sim = sim * active_mask.unsqueeze(0)

        return head_logits + self.gamma * sim

    def reset(self):
        """Reset all prototypes (for episodic mode)."""
        self.prototypes.zero_()
        self.counts.zero_()
