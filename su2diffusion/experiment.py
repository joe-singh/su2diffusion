from dataclasses import dataclass

import torch

from .data import DataConfig, centers_for_config, sample_clean
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
        q_gen_det = sample_reverse(
            model,
            config.schedule,
            n_samples=config.sample_count,
            eta=config.deterministic_eta,
            device=device,
        )
        q_gen_sto = sample_reverse(
            model,
            config.schedule,
            n_samples=config.sample_count,
            eta=config.eta,
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
    )
