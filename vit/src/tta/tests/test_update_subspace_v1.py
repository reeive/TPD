import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.tta.update_subspace import UpdateSubspaceManager


def test_update_subspace_recovers_low_rank_basis():
    manager = UpdateSubspaceManager(rank=2, window_size=5, min_history=2)
    updates = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0]),
    ]
    for update in updates:
        manager.append(update)

    state = manager.fit_basis()
    assert state.ready
    assert state.basis.shape == (3, 2)

    y, residual = manager.project(torch.tensor([2.0, 1.0, 0.0]), state.basis)
    recon = manager.reconstruct(y, residual, state.basis)
    assert torch.allclose(recon, torch.tensor([2.0, 1.0, 0.0]), atol=1e-5)
    assert residual.norm().item() < 1e-5


def test_update_subspace_state_dict_roundtrip():
    manager = UpdateSubspaceManager(rank=2, window_size=4, min_history=2)
    manager.append(torch.tensor([1.0, 0.0]))
    manager.append(torch.tensor([0.5, 0.5]))
    saved = manager.state_dict()

    clone = UpdateSubspaceManager(rank=1, window_size=2, min_history=1)
    clone.load_state_dict(saved)
    assert clone.history_length() == 2
    state = clone.fit_basis()
    assert state.ready
