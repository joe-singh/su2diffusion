from dataclasses import dataclass

import torch

from .data import DataConfig, centers_for_config, sample_balanced_labels, sample_clean
from .diagnostics import SampleDiagnostics, diagnose_samples
from .device import get_default_device
from .diffusion import DiffusionSchedule
from .sampling import sample_reverse
from .quaternion import sample_haar
from .train import TrainConfig, train_heat_kernel_model


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    schedule: DiffusionSchedule
    train: TrainConfig
    data: DataConfig = DataConfig()
    sample_count: int = 5000
    reference_count: int = 5000
    eta: float = 0.7
    deterministic_eta: float = 0.0
    conditional_sampling: bool = False


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    model: torch.nn.Module
    losses: list[float]
    clean_reference: torch.Tensor
    haar_reference: torch.Tensor
    generated_deterministic: torch.Tensor
    generated_stochastic: torch.Tensor
    diagnostics: dict[str, SampleDiagnostics]
    deterministic_labels: torch.Tensor | None = None
    stochastic_labels: torch.Tensor | None = None


def get_experiment_config(name: str) -> ExperimentConfig:
    configs = {
        "smoke": ExperimentConfig(
            name="smoke",
            schedule=DiffusionSchedule(T=30, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=TrainConfig(batch_size=256, num_steps=20, hidden=64, n_terms=16),
            data=DataConfig(kind="blobs", sigma_data=0.18),
            sample_count=512,
            reference_count=512,
            eta=0.7,
        ),
        "smoke-gates": ExperimentConfig(
            name="smoke-gates",
            schedule=DiffusionSchedule(T=30, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=TrainConfig(batch_size=256, num_steps=20, hidden=64, n_terms=16),
            data=DataConfig(kind="gates", sigma_data=0.12, label_strategy="balanced"),
            sample_count=512,
            reference_count=512,
            eta=0.7,
        ),
        "smoke-gates-cond": ExperimentConfig(
            name="smoke-gates-cond",
            schedule=DiffusionSchedule(T=30, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=TrainConfig(batch_size=256, num_steps=20, hidden=64, n_terms=16, conditional=True),
            data=DataConfig(kind="gates", sigma_data=0.12, label_strategy="balanced"),
            sample_count=512,
            reference_count=512,
            eta=0.7,
            conditional_sampling=True,
        ),
        "smoke-cosine": ExperimentConfig(
            name="smoke-cosine",
            schedule=DiffusionSchedule(T=30, beta_start=1e-4, beta_end=0.005, kind="cosine"),
            train=TrainConfig(batch_size=256, num_steps=20, hidden=64, n_terms=16),
            data=DataConfig(kind="blobs", sigma_data=0.18),
            sample_count=512,
            reference_count=512,
            eta=0.7,
        ),
        "medium": ExperimentConfig(
            name="medium",
            schedule=DiffusionSchedule(T=100, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=TrainConfig(batch_size=1024, num_steps=500, hidden=256, n_terms=64),
            data=DataConfig(kind="blobs", sigma_data=0.18),
            sample_count=2000,
            reference_count=2000,
            eta=0.7,
        ),
        "medium-gates": ExperimentConfig(
            name="medium-gates",
            schedule=DiffusionSchedule(T=100, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=TrainConfig(batch_size=1024, num_steps=500, hidden=256, n_terms=64),
            data=DataConfig(kind="gates", sigma_data=0.12, label_strategy="balanced"),
            sample_count=2000,
            reference_count=2000,
            eta=0.7,
        ),
        "medium-gates-cond": ExperimentConfig(
            name="medium-gates-cond",
            schedule=DiffusionSchedule(T=100, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=TrainConfig(batch_size=1024, num_steps=500, hidden=256, n_terms=64, conditional=True),
            data=DataConfig(kind="gates", sigma_data=0.12, label_strategy="balanced"),
            sample_count=2000,
            reference_count=2000,
            eta=0.7,
            conditional_sampling=True,
        ),
        "medium-cosine": ExperimentConfig(
            name="medium-cosine",
            schedule=DiffusionSchedule(T=100, beta_start=1e-4, beta_end=0.005, kind="cosine"),
            train=TrainConfig(batch_size=1024, num_steps=500, hidden=256, n_terms=64),
            data=DataConfig(kind="blobs", sigma_data=0.18),
            sample_count=2000,
            reference_count=2000,
            eta=0.7,
        ),
        "baseline": ExperimentConfig(
            name="baseline",
            schedule=DiffusionSchedule(T=200, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=TrainConfig(batch_size=2048, num_steps=2000, hidden=512, n_terms=128),
            data=DataConfig(kind="blobs", sigma_data=0.18),
            sample_count=5000,
            reference_count=5000,
            eta=0.7,
        ),
        "baseline-gates": ExperimentConfig(
            name="baseline-gates",
            schedule=DiffusionSchedule(T=200, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=TrainConfig(batch_size=2048, num_steps=2000, hidden=512, n_terms=128),
            data=DataConfig(kind="gates", sigma_data=0.12, label_strategy="balanced"),
            sample_count=5000,
            reference_count=5000,
            eta=0.7,
        ),
        "baseline-gates-cond": ExperimentConfig(
            name="baseline-gates-cond",
            schedule=DiffusionSchedule(T=200, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=TrainConfig(batch_size=2048, num_steps=2000, hidden=512, n_terms=128, conditional=True),
            data=DataConfig(kind="gates", sigma_data=0.12, label_strategy="balanced"),
            sample_count=5000,
            reference_count=5000,
            eta=0.7,
            conditional_sampling=True,
        ),
        "baseline-cosine": ExperimentConfig(
            name="baseline-cosine",
            schedule=DiffusionSchedule(T=200, beta_start=1e-4, beta_end=0.005, kind="cosine"),
            train=TrainConfig(batch_size=2048, num_steps=2000, hidden=512, n_terms=128),
            data=DataConfig(kind="blobs", sigma_data=0.18),
            sample_count=5000,
            reference_count=5000,
            eta=0.7,
        ),
    }

    try:
        return configs[name]
    except KeyError as exc:
        valid = ", ".join(sorted(configs))
        raise ValueError(f"Unknown experiment config {name!r}. Valid configs: {valid}") from exc


def run_experiment(
    config: ExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> ExperimentResult:
    if isinstance(config, str):
        config = get_experiment_config(config)

    _validate_experiment_config(config)

    device = torch.device(device) if device is not None else get_default_device()
    centers = centers_for_config(config.data, device=device)

    model, losses = train_heat_kernel_model(
        train_config=config.train,
        schedule=config.schedule,
        blob_config=config.data,
        device=device,
        show_progress=show_progress,
    )

    with torch.no_grad():
        q_clean, _ = sample_clean(
            config.reference_count,
            centers=centers,
            config=config.data,
        )
        q_haar = sample_haar(config.reference_count, device=device)
        deterministic_labels = _sample_condition_labels(config, centers.shape[0], device=device)
        stochastic_labels = _sample_condition_labels(config, centers.shape[0], device=device)
        q_gen_det = sample_reverse(
            model,
            config.schedule,
            n_samples=config.sample_count,
            eta=config.deterministic_eta,
            labels=deterministic_labels,
            device=device,
        )
        q_gen_sto = sample_reverse(
            model,
            config.schedule,
            n_samples=config.sample_count,
            eta=config.eta,
            labels=stochastic_labels,
            device=device,
        )

    diagnostics = {
        "deterministic": diagnose_samples(q_gen_det, q_clean, q_haar, centers=centers),
        "stochastic": diagnose_samples(q_gen_sto, q_clean, q_haar, centers=centers),
    }

    return ExperimentResult(
        config=config,
        model=model,
        losses=losses,
        clean_reference=q_clean,
        haar_reference=q_haar,
        generated_deterministic=q_gen_det,
        generated_stochastic=q_gen_sto,
        diagnostics=diagnostics,
        deterministic_labels=deterministic_labels.detach().cpu() if deterministic_labels is not None else None,
        stochastic_labels=stochastic_labels.detach().cpu() if stochastic_labels is not None else None,
    )


def _sample_condition_labels(
    config: ExperimentConfig,
    n_centers: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not config.conditional_sampling:
        return None
    return sample_balanced_labels(
        batch_size=config.sample_count,
        n_centers=n_centers,
        device=device,
    )


def _validate_experiment_config(config: ExperimentConfig) -> None:
    if config.train.conditional and not config.conditional_sampling:
        raise ValueError("Conditional training requires conditional_sampling=True for experiment sampling")
    if config.conditional_sampling and not config.train.conditional:
        raise ValueError("conditional_sampling=True requires train.conditional=True")
