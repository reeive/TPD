"""
Plugin: Prompt Quality Tracker

Tracks per-prompt quality using entropy as a proxy for correctness.
Low-quality prompts (consistently high post-adaptation entropy) get:
  - Stronger anchor regularization (higher λ multiplier)
  - Lower selection priority (penalty added to entropy-based selection)

Usage:
    tracker = PromptQualityTracker(num_prompts=12)
    # After adaptation:
    tracker.update(selected_idx, post_entropy)
    # Before adaptation:
    lambda_mult = tracker.get_lambda_multiplier(selected_idx)
    # For selection override:
    adjusted_ent = tracker.adjust_selection_entropy(per_prompt_entropies)
"""

import numpy as np
from typing import List, Optional


class PromptQualityTracker:
    """EMA-based per-prompt quality tracking."""

    def __init__(
        self,
        num_prompts: int = 12,
        ema_alpha: float = 0.95,
        window_size: int = 50,
        quality_lambda_scale: float = 2.0,
        selection_penalty_scale: float = 0.5,
    ):
        """
        Args:
            num_prompts: Number of prompts to track.
            ema_alpha: EMA decay for entropy tracking (higher = slower update).
            window_size: Minimum samples before quality estimates are trusted.
            quality_lambda_scale: Max multiplier on λ for worst prompt.
                λ_eff = λ × (1 + (quality_lambda_scale - 1) × badness)
                where badness ∈ [0, 1].
            selection_penalty_scale: Entropy penalty for bad prompts during
                selection.  Added to raw entropy before argmin.
        """
        self.num_prompts = num_prompts
        self.ema_alpha = ema_alpha
        self.window_size = window_size
        self.quality_lambda_scale = quality_lambda_scale
        self.selection_penalty_scale = selection_penalty_scale

        self._ent_ema = [0.5] * num_prompts
        self._usage_count = [0] * num_prompts
        self._total_samples = 0

    def update(self, prompt_idx: int, post_entropy: float):
        """Record observed entropy after adapting with this prompt."""
        self._usage_count[prompt_idx] += 1
        alpha = self.ema_alpha
        self._ent_ema[prompt_idx] = (
            alpha * self._ent_ema[prompt_idx] + (1 - alpha) * post_entropy
        )
        self._total_samples += 1

    def _badness(self, prompt_idx: int) -> float:
        """Compute normalized badness ∈ [0, 1] for a prompt.

        0 = best prompt, 1 = worst prompt.  Based on relative EMA entropy
        compared to the best and worst prompts.
        """
        if self._total_samples < self.window_size:
            return 0.0

        active = [
            (i, self._ent_ema[i])
            for i in range(self.num_prompts)
            if self._usage_count[i] >= 3
        ]
        if len(active) < 2:
            return 0.0

        ents = [e for _, e in active]
        lo, hi = min(ents), max(ents)
        if hi - lo < 1e-6:
            return 0.0

        my_ent = self._ent_ema[prompt_idx]
        return max(0.0, min(1.0, (my_ent - lo) / (hi - lo)))

    def get_lambda_multiplier(self, prompt_idx: int) -> float:
        """Get anchor λ multiplier for this prompt.

        Returns 1.0 for the best prompt, up to quality_lambda_scale for worst.
        """
        b = self._badness(prompt_idx)
        return 1.0 + (self.quality_lambda_scale - 1.0) * b

    def adjust_selection_entropy(
        self, per_prompt_entropies: List[float]
    ) -> List[float]:
        """Add quality-based penalty to entropy for prompt selection.

        Worse prompts get higher adjusted entropy → less likely to be selected.
        """
        if self._total_samples < self.window_size:
            return per_prompt_entropies

        adjusted = []
        for i, ent in enumerate(per_prompt_entropies):
            penalty = self.selection_penalty_scale * self._badness(i)
            adjusted.append(ent + penalty)
        return adjusted

    @property
    def stats(self):
        return {
            "ent_ema": {i: round(e, 4) for i, e in enumerate(self._ent_ema)},
            "usage": dict(enumerate(self._usage_count)),
            "total_samples": self._total_samples,
        }

    def summary(self) -> str:
        """One-line summary of prompt quality ranking."""
        ranked = sorted(
            range(self.num_prompts),
            key=lambda i: self._ent_ema[i],
        )
        parts = []
        for i in ranked[:5]:
            parts.append(f"p{i}={self._ent_ema[i]:.2f}({self._usage_count[i]})")
        return " ".join(parts)
