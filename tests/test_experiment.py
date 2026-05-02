import pytest

from su2diffusion import (
    DiffusionSchedule,
    ExperimentConfig,
    TrainConfig,
    get_experiment_config,
    run_experiment,
)


def test_get_experiment_config_returns_named_config():
    config = get_experiment_config("smoke")

    assert config.name == "smoke"
    assert config.sample_count == 512
    assert config.reference_count == 512


def test_get_experiment_config_returns_cosine_variant():
    config = get_experiment_config("smoke-cosine")

    assert config.name == "smoke-cosine"
    assert config.schedule.kind == "cosine"
    assert config.deterministic_eta == 0.0


def test_get_experiment_config_returns_gate_variant():
    config = get_experiment_config("smoke-gates")

    assert config.name == "smoke-gates"
    assert config.data.kind == "gates"
    assert config.data.sigma_data == 0.12


def test_get_experiment_config_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown experiment config"):
        get_experiment_config("missing")


def test_run_experiment_returns_expected_shapes():
    config = ExperimentConfig(
        name="test",
        schedule=DiffusionSchedule(T=3, beta_start=1e-4, beta_end=0.005),
        train=TrainConfig(batch_size=4, num_steps=1, hidden=8, n_terms=4),
        sample_count=6,
        reference_count=7,
        eta=0.5,
    )

    result = run_experiment(config, device="cpu", show_progress=False)

    assert len(result.losses) == 1
    assert result.clean_reference.shape == (7, 4)
    assert result.haar_reference.shape == (7, 4)
    assert result.generated_deterministic.shape == (6, 4)
    assert result.generated_stochastic.shape == (6, 4)
    assert sorted(result.diagnostics) == ["deterministic", "stochastic"]
    assert result.diagnostics["deterministic"].distance_to_clean_w1 >= 0.0
