import torch

from su2diffusion.quaternion import q_exp, q_inv, q_log, q_mul, q_normalize, sample_haar


def test_quaternion_inverse_is_identity():
    q = sample_haar(32, device="cpu")
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(q)

    actual = q_mul(q, q_inv(q))

    assert torch.allclose(actual, identity, atol=1e-5)


def test_exp_log_round_trip_near_identity():
    v = 0.2 * torch.randn(32, 3)

    actual = q_log(q_exp(v))

    assert torch.allclose(actual, v, atol=1e-5)


def test_normalize_returns_unit_quaternions():
    q = torch.randn(32, 4)

    actual = q_normalize(q)

    assert torch.allclose(actual.norm(dim=-1), torch.ones(32), atol=1e-6)
