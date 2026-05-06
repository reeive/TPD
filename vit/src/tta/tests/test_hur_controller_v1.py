import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.tta.hur_controller import HankelUpdateRouter


def test_hur_split_is_exactly_additive():
    router = HankelUpdateRouter(beta_c=0.2, beta_n=0.1, kappa=2.0, eps=1e-6)
    current = torch.tensor([1.0, -0.5])
    previous = torch.tensor([0.8, 0.4])
    coefficients = torch.tensor([0.9, -0.7])

    split = router.split(current, previous, coefficients)
    assert torch.allclose(
        split.persistent + split.oscillatory + split.noise,
        current,
        atol=1e-6,
    )


def test_hur_route_matches_weighted_recombination():
    router = HankelUpdateRouter(beta_c=0.3, beta_n=0.0)
    current = torch.tensor([1.0, 0.2])
    previous = torch.tensor([0.9, -0.2])
    coefficients = torch.tensor([0.8, -0.6])
    basis = torch.eye(2)
    residual = torch.tensor([0.1, -0.1])

    split = router.split(current, previous, coefficients)
    routed = router.route(split, basis, residual)

    expected = routed.persistent + 0.3 * routed.oscillatory
    assert torch.allclose(routed.routed, expected, atol=1e-6)
