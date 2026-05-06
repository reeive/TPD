import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.tta.tpd_runtime import TPDRuntime


class DummyPromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.zeros(3))


def test_tpd_runtime_uses_applied_updates_in_history():
    model = DummyPromptModel()
    runtime = TPDRuntime(
        model,
        base_lr=1.0,
        subspace_rank=1,
        window_size=4,
        min_history=2,
        akc_mode="diagonal",
        rho_threshold=10.0,
        hur_beta_c=0.0,
        hur_beta_n=0.0,
    )

    runtime.step(raw_update=torch.tensor([1.0, 0.0, 0.0]))
    runtime.step(raw_update=torch.tensor([1.0, 0.0, 0.0]))
    result = runtime.step(raw_update=torch.tensor([1.0, 0.0, 0.0]))

    history = runtime.subspace.history_tensor()
    assert history is not None
    assert history.shape[0] == 3
    assert torch.allclose(model.prompt_embeddings.detach(), history.sum(dim=0), atol=1e-6)
    assert not torch.allclose(history[-1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert result.u_applied_norm <= result.u_raw_norm + 1e-6


def test_tpd_runtime_rolls_back_to_last_stable_checkpoint():
    model = DummyPromptModel()
    runtime = TPDRuntime(
        model,
        base_lr=1.0,
        subspace_rank=1,
        window_size=4,
        min_history=2,
        akc_mode="diagonal",
        rho_threshold=0.5,
        rollback_patience=1,
        hur_beta_c=0.0,
        hur_beta_n=0.0,
    )

    runtime.step(raw_update=torch.tensor([0.2, 0.0, 0.0]))
    runtime.step(raw_update=torch.tensor([0.2, 0.0, 0.0]))
    checkpoint_state = model.prompt_embeddings.detach().clone()

    result = runtime.step(raw_update=torch.tensor([2.0, 0.0, 0.0]))
    assert result.rollback is True
    assert torch.allclose(model.prompt_embeddings.detach(), checkpoint_state, atol=1e-6)
