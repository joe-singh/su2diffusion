import pytest
import torch

from su2diffusion.model import SU2Denoiser


def test_unconditional_denoiser_accepts_no_labels():
    model = SU2Denoiser(T=4, hidden=8)
    q = torch.randn(3, 4)
    t_idx = torch.ones(3, dtype=torch.long)

    actual = model(q, t_idx)

    assert actual.shape == (3, 3)


def test_conditional_denoiser_requires_labels():
    model = SU2Denoiser(T=4, hidden=8, num_labels=7, label_dim=4)
    q = torch.randn(3, 4)
    t_idx = torch.ones(3, dtype=torch.long)

    with pytest.raises(ValueError, match="requires labels"):
        model(q, t_idx)


def test_conditional_denoiser_accepts_labels():
    model = SU2Denoiser(T=4, hidden=8, num_labels=7, label_dim=4)
    q = torch.randn(3, 4)
    t_idx = torch.ones(3, dtype=torch.long)
    labels = torch.tensor([0, 1, 2])

    actual = model(q, t_idx, labels=labels)

    assert actual.shape == (3, 3)
