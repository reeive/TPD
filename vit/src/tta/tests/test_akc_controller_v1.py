import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.tta.akc_controller import AdaptiveKoopmanController


def _simulate_history(operator: torch.Tensor, steps: int = 8):
    state = torch.tensor([1.0, -0.2], dtype=torch.float32)
    history = [state]
    for _ in range(steps - 1):
        state = operator @ state
        history.append(state)
    return torch.stack(history, dim=0)


def test_akc_auto_switch_prefers_full_for_coupled_dynamics():
    operator = torch.tensor([[0.85, 0.25], [0.0, 0.7]], dtype=torch.float32)
    history = _simulate_history(operator, steps=8)

    controller = AdaptiveKoopmanController(
        mode="auto",
        rho_threshold=1.0,
        full_condition_threshold=1e6,
        full_improvement_margin=0.0,
        min_mode_dwell=0,
    )
    decision = controller.estimate(history)
    assert decision.mode == "full"
    assert decision.full_fit is not None
    assert decision.diag_fit is not None


def test_akc_auto_falls_back_to_diagonal_with_short_history():
    operator = torch.tensor([[0.85, 0.25], [0.0, 0.7]], dtype=torch.float32)
    history = _simulate_history(operator, steps=2)

    controller = AdaptiveKoopmanController(
        mode="auto",
        rho_threshold=1.0,
        min_mode_dwell=0,
    )
    decision = controller.estimate(history)
    assert decision.mode == "diagonal"


def test_akc_triggers_rollback_after_persistent_instability():
    operator = torch.tensor([[1.3, 0.0], [0.0, 1.1]], dtype=torch.float32)
    history = _simulate_history(operator, steps=6)

    controller = AdaptiveKoopmanController(
        mode="diagonal",
        rho_threshold=1.0,
        rollback_patience=2,
    )
    first = controller.estimate(history)
    second = controller.estimate(history)

    assert first.rollback is False
    assert second.rollback is True
    assert second.scale < 1.0
