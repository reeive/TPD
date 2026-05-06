"""
Plugin: Adaptive Anchor Lambda

Adjusts anchor regularization strength λ over time.  Early samples benefit
from more exploration (lower λ); later samples need stronger anchoring
to prevent accumulated drift.

Supports multiple schedules:
  - 'linear':  λ(t) = λ_base × (1 + t/T)
  - 'sqrt':    λ(t) = λ_base × (1 + sqrt(t/T))
  - 'step':    λ(t) = λ_base × 2^(t // step_size)

Usage:
    scheduler = AdaptiveLambda(base_lambda=5.0, schedule='linear', T=5000)
    # Each sample:
    current_lambda = scheduler.step()
"""


class AdaptiveLambda:
    """Time-adaptive anchor regularization strength."""

    def __init__(
        self,
        base_lambda: float = 5.0,
        schedule: str = "linear",
        T: float = 5000.0,
        max_lambda: float = 50.0,
        step_size: int = 500,
    ):
        """
        Args:
            base_lambda: Starting λ value.
            schedule: One of 'constant', 'linear', 'sqrt', 'step'.
            T: Time constant (used by linear and sqrt schedules).
            max_lambda: Hard upper bound on λ.
            step_size: Step interval for 'step' schedule.
        """
        self.base_lambda = base_lambda
        self.schedule = schedule
        self.T = max(T, 1.0)
        self.max_lambda = max_lambda
        self.step_size = max(step_size, 1)
        self._t = 0
        self._history = []

    def get_lambda(self) -> float:
        """Compute current λ without advancing time."""
        t = self._t
        if self.schedule == "constant":
            lam = self.base_lambda
        elif self.schedule == "linear":
            lam = self.base_lambda * (1.0 + t / self.T)
        elif self.schedule == "sqrt":
            lam = self.base_lambda * (1.0 + (t / self.T) ** 0.5)
        elif self.schedule == "step":
            doublings = t // self.step_size
            lam = self.base_lambda * (2.0 ** doublings)
        else:
            lam = self.base_lambda
        return min(lam, self.max_lambda)

    def step(self) -> float:
        """Advance time by 1 and return current λ."""
        lam = self.get_lambda()
        self._history.append(lam)
        self._t += 1
        return lam

    def reset(self):
        self._t = 0
        self._history = []

    @property
    def stats(self):
        if not self._history:
            return {"lambda_current": self.base_lambda, "lambda_mean": self.base_lambda}
        return {
            "lambda_current": self._history[-1],
            "lambda_mean": sum(self._history) / len(self._history),
            "lambda_min": min(self._history),
            "lambda_max": max(self._history),
        }
