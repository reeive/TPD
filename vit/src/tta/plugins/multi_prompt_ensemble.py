"""
Plugin: Multi-Prompt Ensemble Voting

For high-entropy (uncertain) samples, instead of relying on entropy-based
single-prompt selection, aggregate predictions from ALL prompts via logit
averaging.  This avoids selecting a poor prompt when entropy is unreliable.

Usage:
    ensemble = MultiPromptEnsemble(threshold=1.5)
    # In the TTA loop:
    if ensemble.should_ensemble(raw_entropy):
        pred = ensemble.ensemble_predict(model, image, num_prompts)
    else:
        pred = single_prompt_predict(...)
"""

import torch
import numpy as np
from typing import Optional


class MultiPromptEnsemble:
    """Ensemble voting over multiple prompts for uncertain samples."""

    def __init__(
        self,
        entropy_threshold: float = 1.5,
        top_k: int = 0,
        temperature: float = 1.0,
    ):
        """
        Args:
            entropy_threshold: Use ensemble when raw_entropy > this value.
            top_k: If > 0, only average logits from top-k lowest-entropy
                   prompts instead of all.  0 = use all prompts.
            temperature: Logit temperature before averaging (1.0 = no change).
        """
        self.entropy_threshold = entropy_threshold
        self.top_k = top_k
        self.temperature = temperature
        self._ensemble_count = 0
        self._single_count = 0

    def should_ensemble(self, entropy: float) -> bool:
        return entropy > self.entropy_threshold

    @torch.no_grad()
    def ensemble_predict(
        self,
        model,
        image: torch.Tensor,
        num_prompts: int,
        selected_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward all prompts and aggregate logits.

        Args:
            model: Multi-prompt CLIP model with forward(image, p_idx).
            image: Single test image (1, C, H, W) — no augmentation.
            num_prompts: Total number of prompts.
            selected_idx: If provided, also return this prompt's raw logits
                          for compatibility (unused for prediction).

        Returns:
            ensembled_logits: (1, num_classes) aggregated logits.
        """
        all_logits = []
        all_ent = []

        for p_idx in range(num_prompts):
            with torch.cuda.amp.autocast():
                logits = model(image, p_idx + 1)  # (1, num_classes)
            if logits.dim() == 3:
                logits = logits.mean(1)
            logits = logits / self.temperature
            ent = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1).mean()
            all_logits.append(logits)
            all_ent.append(ent.item())

        if self.top_k > 0 and self.top_k < num_prompts:
            topk_idx = np.argsort(all_ent)[:self.top_k]
            selected = [all_logits[i] for i in topk_idx]
        else:
            selected = all_logits

        ensembled = torch.stack(selected).mean(dim=0)
        self._ensemble_count += 1
        return ensembled

    def record_single(self):
        self._single_count += 1

    @property
    def stats(self):
        total = self._ensemble_count + self._single_count
        return {
            "ensemble_count": self._ensemble_count,
            "single_count": self._single_count,
            "ensemble_ratio": self._ensemble_count / max(total, 1),
        }
