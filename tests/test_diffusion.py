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


def test_cosine_schedule_has_positive_betas_and_expected_terminal_variance():
    schedule = DiffusionSchedule(T=8, beta_start=1e-4, beta_end=0.005, kind="cosine")

    betas, sigma2, sigmas = schedule.tensors(device="cpu")

    assert betas.shape == (8,)
    assert (betas > 0).all()
    assert torch.allclose(sigma2[-1], torch.tensor(8 * 0.005), atol=1e-6)
    assert torch.allclose(sigmas, sigma2.sqrt())


def test_unknown_schedule_kind_raises_value_error():
    schedule = DiffusionSchedule(kind="missing")

    try:
        schedule.tensors(device="cpu")
    except ValueError as exc:
        assert "Unknown diffusion schedule kind" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
