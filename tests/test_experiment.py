import pytest

from su2diffusion import (
    DiffusionSchedule,
    ExperimentConfig,
    TrainConfig,
    get_experiment_config,
    resample_experiment,
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
    assert config.data.label_strategy == "balanced"


def test_get_experiment_config_returns_conditional_gate_variant():
    config = get_experiment_config("smoke-gates-cond")

    assert config.name == "smoke-gates-cond"
    assert config.data.kind == "gates"
    assert config.train.conditional is True
    assert config.conditional_sampling is True
    assert config.eta == 1.0


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
    assert result.deterministic_labels is None
    assert result.stochastic_labels is None


def test_run_conditional_experiment_returns_sampling_labels():
    config = ExperimentConfig(
        name="test-cond",
        schedule=DiffusionSchedule(T=3, beta_start=1e-4, beta_end=0.005),
        train=TrainConfig(batch_size=4, num_steps=1, hidden=8, n_terms=4, conditional=True, label_dim=4),
        sample_count=6,
        reference_count=7,
        eta=0.5,
        conditional_sampling=True,
    )

    result = run_experiment(config, device="cpu", show_progress=False)

    assert result.generated_deterministic.shape == (6, 4)
    assert result.deterministic_labels is not None
    assert result.stochastic_labels is not None
    assert result.deterministic_labels.shape == (6,)
    assert result.stochastic_labels.shape == (6,)


def test_resample_experiment_reuses_model_for_eta_sweep():
    config = ExperimentConfig(
        name="test-resample",
        schedule=DiffusionSchedule(T=3, beta_start=1e-4, beta_end=0.005),
        train=TrainConfig(batch_size=4, num_steps=1, hidden=8, n_terms=4, conditional=True, label_dim=4),
        sample_count=6,
        reference_count=7,
        eta=0.5,
        conditional_sampling=True,
    )
    result = run_experiment(config, device="cpu", show_progress=False)

    sweep = resample_experiment(result, etas=[0.0, 0.5], device="cpu")

    assert sorted(sweep) == ["eta=0", "eta=0.5"]
    assert sweep["eta=0"].eta == 0.0
    assert sweep["eta=0"].generated.shape == (6, 4)
    assert sweep["eta=0"].labels is not None
    assert sweep["eta=0"].diagnostics.distance_to_clean_w1 >= 0.0


def test_run_experiment_rejects_conditional_training_without_conditional_sampling():
    config = ExperimentConfig(
        name="bad-cond",
        schedule=DiffusionSchedule(T=3, beta_start=1e-4, beta_end=0.005),
        train=TrainConfig(batch_size=4, num_steps=1, hidden=8, n_terms=4, conditional=True),
        sample_count=6,
        reference_count=7,
        conditional_sampling=False,
    )

    with pytest.raises(ValueError, match="Conditional training requires conditional_sampling"):
        run_experiment(config, device="cpu", show_progress=False)


def test_run_experiment_rejects_conditional_sampling_without_conditional_training():
    config = ExperimentConfig(
        name="bad-sampling",
        schedule=DiffusionSchedule(T=3, beta_start=1e-4, beta_end=0.005),
        train=TrainConfig(batch_size=4, num_steps=1, hidden=8, n_terms=4, conditional=False),
        sample_count=6,
        reference_count=7,
        conditional_sampling=True,
    )

    with pytest.raises(ValueError, match="conditional_sampling=True requires train.conditional=True"):
        run_experiment(config, device="cpu", show_progress=False)
