import torch

from su2diffusion.data import (
    DataConfig,
    center_names_for_config,
    centers_for_config,
    gate_centers,
    sample_balanced_labels,
    sample_clean,
)


def test_gate_centers_are_unit_quaternions():
    centers = gate_centers(device="cpu")

    assert centers.shape == (7, 4)
    assert torch.allclose(centers.norm(dim=-1), torch.ones(7), atol=1e-6)


def test_gate_center_names_match_gate_centers():
    config = DataConfig(kind="gates")

    assert center_names_for_config(config) == ["I", "X", "Y", "Z", "sqrt(X)", "sqrt(Y)", "sqrt(Z)"]
    assert len(center_names_for_config(config)) == gate_centers(device="cpu").shape[0]


def test_sample_clean_gate_data_shapes():
    config = DataConfig(kind="gates", sigma_data=0.12)
    centers = centers_for_config(config, device="cpu")

    q, labels = sample_clean(32, centers=centers, config=config)

    assert q.shape == (32, 4)
    assert labels.shape == (32,)
    assert labels.max().item() < centers.shape[0]
    assert torch.allclose(q.norm(dim=-1), torch.ones(32), atol=1e-5)


def test_balanced_label_strategy_has_nearly_equal_center_counts():
    config = DataConfig(kind="gates", sigma_data=0.12, label_strategy="balanced")
    centers = centers_for_config(config, device="cpu")

    _, labels = sample_clean(32, centers=centers, config=config)
    counts = torch.bincount(labels.cpu(), minlength=centers.shape[0])

    assert counts.max().item() - counts.min().item() <= 1


def test_sample_balanced_labels_has_nearly_equal_counts():
    labels = sample_balanced_labels(32, n_centers=7, device="cpu")
    counts = torch.bincount(labels.cpu(), minlength=7)

    assert counts.max().item() - counts.min().item() <= 1


def test_unknown_label_strategy_raises_value_error():
    config = DataConfig(kind="gates", label_strategy="missing")

    try:
        sample_clean(8, config=config, device="cpu")
    except ValueError as exc:
        assert "Unknown label strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_unknown_data_kind_raises_value_error():
    config = DataConfig(kind="missing")

    try:
        centers_for_config(config, device="cpu")
    except ValueError as exc:
        assert "Unknown data kind" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
