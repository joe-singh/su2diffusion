import torch

from su2diffusion.data import DataConfig, centers_for_config, gate_centers, sample_clean


def test_gate_centers_are_unit_quaternions():
    centers = gate_centers(device="cpu")

    assert centers.shape == (7, 4)
    assert torch.allclose(centers.norm(dim=-1), torch.ones(7), atol=1e-6)


def test_sample_clean_gate_data_shapes():
    config = DataConfig(kind="gates", sigma_data=0.12)
    centers = centers_for_config(config, device="cpu")

    q, labels = sample_clean(32, centers=centers, config=config)

    assert q.shape == (32, 4)
    assert labels.shape == (32,)
    assert labels.max().item() < centers.shape[0]
    assert torch.allclose(q.norm(dim=-1), torch.ones(32), atol=1e-5)


def test_unknown_data_kind_raises_value_error():
    config = DataConfig(kind="missing")

    try:
        centers_for_config(config, device="cpu")
    except ValueError as exc:
        assert "Unknown data kind" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
