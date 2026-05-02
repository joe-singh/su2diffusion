import torch

from su2diffusion.data import sample_clean_blobs
from su2diffusion.diffusion import DiffusionSchedule, brownian_forward_heat_target


def test_brownian_forward_heat_target_shapes():
    schedule = DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005)
    q0, labels = sample_clean_blobs(8, device="cpu")
    t_idx = torch.randint(1, schedule.T + 1, (8,), device="cpu")

    qt, eps_target = brownian_forward_heat_target(q0, t_idx, schedule=schedule, n_terms=16)

    assert labels.shape == (8,)
    assert qt.shape == (8, 4)
    assert eps_target.shape == (8, 3)
    assert torch.allclose(qt.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.isfinite(eps_target).all()
