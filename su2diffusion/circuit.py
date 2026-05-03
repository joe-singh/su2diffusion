from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import DataConfig, center_names_for_config, centers_for_config
from .device import get_default_device
from .diffusion import DiffusionSchedule, brownian_forward_heat_target
from .model import CircuitDenoiser, TargetConditionedCircuitDenoiser
from .quaternion import q_exp, q_mul, q_normalize, sample_haar
from .synthesis import (
    HiddenShallowCircuitAggregate,
    HiddenShallowCircuitTarget,
    NearCliffordCircuitBenchmark,
    RefinementResult,
    SynthesisCandidate,
    SynthesisReport,
    compose_two_entangler_local,
    make_near_clifford_two_entangler_circuit_targets,
    quaternion_to_unitary,
    refine_two_entangler_candidate,
    sample_near_clifford_gates,
    summarize_near_clifford_two_entangler_benchmark,
    two_qubit_gate,
    unitary_fidelity,
    unitary_fidelity_batch,
)


@dataclass(frozen=True)
class CircuitTrainConfig:
    batch_size: int = 1024
    num_steps: int = 1000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    hidden: int = 512
    n_terms: int = 128
    seed: int = 0


@dataclass(frozen=True)
class CircuitExperimentConfig:
    name: str
    schedule: DiffusionSchedule
    train: CircuitTrainConfig
    data: DataConfig = DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    sample_count: int = 5000
    eta: float = 0.7
    deterministic_eta: float = 0.0
    n_slots: int = 6


@dataclass
class CircuitExperimentResult:
    config: CircuitExperimentConfig
    model: torch.nn.Module
    losses: list[float]
    generated_deterministic: torch.Tensor
    generated_stochastic: torch.Tensor


@dataclass
class TargetConditionedCircuitExperimentResult:
    config: CircuitExperimentConfig
    model: torch.nn.Module
    losses: list[float]
    generated_deterministic_by_target: torch.Tensor
    generated_stochastic_by_target: torch.Tensor


@dataclass
class SolutionStackDatasetResult:
    stacks: torch.Tensor
    fidelities: torch.Tensor
    targets: list[HiddenShallowCircuitTarget]
    refinements: list[RefinementResult]


@dataclass
class TargetConditionedOverfitResult:
    target_unitaries: torch.Tensor
    solution_stacks: torch.Tensor
    reports: list[SynthesisReport]
    losses: list[float]
    generated_stochastic_by_target: torch.Tensor


def get_circuit_experiment_config(name: str) -> CircuitExperimentConfig:
    configs = {
        "smoke-circuit-near-clifford": CircuitExperimentConfig(
            name="smoke-circuit-near-clifford",
            schedule=DiffusionSchedule(T=30, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=CircuitTrainConfig(batch_size=128, num_steps=20, hidden=64, n_terms=16),
            sample_count=256,
            eta=0.7,
        ),
        "medium-circuit-near-clifford": CircuitExperimentConfig(
            name="medium-circuit-near-clifford",
            schedule=DiffusionSchedule(T=100, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=CircuitTrainConfig(batch_size=512, num_steps=500, hidden=256, n_terms=64),
            sample_count=2000,
            eta=0.7,
        ),
        "baseline-circuit-near-clifford": CircuitExperimentConfig(
            name="baseline-circuit-near-clifford",
            schedule=DiffusionSchedule(T=200, beta_start=1e-4, beta_end=0.005, kind="linear"),
            train=CircuitTrainConfig(batch_size=1024, num_steps=2000, hidden=512, n_terms=128),
            sample_count=5000,
            eta=0.7,
        ),
    }
    try:
        return configs[name]
    except KeyError as exc:
        valid = ", ".join(sorted(configs))
        raise ValueError(f"Unknown circuit experiment config {name!r}. Valid configs: {valid}") from exc


def sample_near_clifford_circuit_stacks(
    batch_size: int,
    centers: torch.Tensor,
    center_names: list[str],
    sigma: float = 0.08,
    n_slots: int = 6,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[tuple[str, ...]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    if centers.shape[0] != len(center_names):
        raise ValueError("center_names must match centers")

    labels = torch.randint(
        low=0,
        high=centers.shape[0],
        size=(batch_size, n_slots),
        device=centers.device,
        generator=generator,
    )
    perturbations = sigma * torch.randn(batch_size, n_slots, 3, device=centers.device, generator=generator)
    q_stack = q_normalize(q_mul(q_exp(perturbations), centers[labels]))
    label_names = [tuple(center_names[int(label)] for label in row.tolist()) for row in labels]
    return q_stack, label_names


def circuit_forward_heat_target(
    q0_stack: torch.Tensor,
    t_idx: torch.Tensor,
    schedule: DiffusionSchedule,
    n_terms: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q0_stack.ndim != 3 or q0_stack.shape[-1] != 4:
        raise ValueError("q0_stack must have shape (batch, n_slots, 4)")

    batch_size, n_slots, _ = q0_stack.shape
    flat_q0 = q0_stack.reshape(batch_size * n_slots, 4)
    flat_t = t_idx.repeat_interleave(n_slots)
    flat_qt, flat_eps = brownian_forward_heat_target(flat_q0, flat_t, schedule=schedule, n_terms=n_terms)
    return flat_qt.reshape(batch_size, n_slots, 4), flat_eps.reshape(batch_size, n_slots, 3)


def target_unitary_features(target_unitaries: torch.Tensor) -> torch.Tensor:
    if target_unitaries.ndim < 2 or target_unitaries.shape[-2:] != (4, 4):
        raise ValueError("target_unitaries must have shape (..., 4, 4)")
    target_unitaries = target_unitaries.to(dtype=torch.complex64)
    return torch.cat(
        [
            target_unitaries.real.reshape(*target_unitaries.shape[:-2], 16),
            target_unitaries.imag.reshape(*target_unitaries.shape[:-2], 16),
        ],
        dim=-1,
    )


def train_circuit_heat_kernel_model(
    train_config: CircuitTrainConfig | None = None,
    schedule: DiffusionSchedule | None = None,
    data_config: DataConfig | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> tuple[CircuitDenoiser, list[float]]:
    train_config = train_config or CircuitTrainConfig()
    schedule = schedule or DiffusionSchedule()
    data_config = data_config or DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    device = torch.device(device) if device is not None else get_default_device()

    torch.manual_seed(train_config.seed)
    centers = centers_for_config(data_config, device=device)
    center_names = center_names_for_config(data_config)

    model = CircuitDenoiser(T=schedule.T, hidden=train_config.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    losses: list[float] = []

    iterator = range(1, train_config.num_steps + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Training circuit heat-kernel target", dynamic_ncols=True)

    for _ in iterator:
        q0_stack, _ = sample_near_clifford_circuit_stacks(
            train_config.batch_size,
            centers=centers,
            center_names=center_names,
            sigma=data_config.sigma_data,
        )
        t_idx = torch.randint(1, schedule.T + 1, (train_config.batch_size,), device=device)

        with torch.no_grad():
            qt_stack, eps_target = circuit_forward_heat_target(
                q0_stack,
                t_idx,
                schedule=schedule,
                n_terms=train_config.n_terms,
            )

        eps_pred = model(qt_stack, t_idx)
        loss = F.mse_loss(eps_pred, eps_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_value = loss.item()
        losses.append(loss_value)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_value:.5f}"})

    return model, losses


def train_circuit_heat_kernel_model_on_stacks(
    solution_stacks: torch.Tensor,
    train_config: CircuitTrainConfig | None = None,
    schedule: DiffusionSchedule | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> tuple[CircuitDenoiser, list[float]]:
    if solution_stacks.ndim != 3 or solution_stacks.shape[1:] != (6, 4):
        raise ValueError("solution_stacks must have shape (n, 6, 4)")
    if solution_stacks.shape[0] == 0:
        raise ValueError("solution_stacks must contain at least one stack")

    train_config = train_config or CircuitTrainConfig()
    schedule = schedule or DiffusionSchedule()
    device = torch.device(device) if device is not None else get_default_device()
    solution_stacks = q_normalize(solution_stacks.to(device=device))

    torch.manual_seed(train_config.seed)
    model = CircuitDenoiser(T=schedule.T, hidden=train_config.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    losses: list[float] = []

    iterator = range(1, train_config.num_steps + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Training solution-stack circuit target", dynamic_ncols=True)

    for _ in iterator:
        rows = torch.randint(
            low=0,
            high=solution_stacks.shape[0],
            size=(train_config.batch_size,),
            device=device,
        )
        q0_stack = solution_stacks[rows]
        t_idx = torch.randint(1, schedule.T + 1, (train_config.batch_size,), device=device)

        with torch.no_grad():
            qt_stack, eps_target = circuit_forward_heat_target(
                q0_stack,
                t_idx,
                schedule=schedule,
                n_terms=train_config.n_terms,
            )

        eps_pred = model(qt_stack, t_idx)
        loss = F.mse_loss(eps_pred, eps_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_value = loss.item()
        losses.append(loss_value)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_value:.5f}"})

    return model, losses


def train_target_conditioned_circuit_heat_kernel_model(
    solution_stacks: torch.Tensor,
    target_unitaries: torch.Tensor,
    train_config: CircuitTrainConfig | None = None,
    schedule: DiffusionSchedule | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> tuple[TargetConditionedCircuitDenoiser, list[float]]:
    if solution_stacks.ndim != 3 or solution_stacks.shape[1:] != (6, 4):
        raise ValueError("solution_stacks must have shape (n, 6, 4)")
    if target_unitaries.ndim != 3 or target_unitaries.shape[1:] != (4, 4):
        raise ValueError("target_unitaries must have shape (n, 4, 4)")
    if solution_stacks.shape[0] == 0:
        raise ValueError("solution_stacks must contain at least one stack")
    if target_unitaries.shape[0] != solution_stacks.shape[0]:
        raise ValueError("target_unitaries must have one target per solution stack")

    train_config = train_config or CircuitTrainConfig()
    schedule = schedule or DiffusionSchedule()
    device = torch.device(device) if device is not None else get_default_device()
    solution_stacks = q_normalize(solution_stacks.to(device=device))
    target_features = target_unitary_features(target_unitaries.to(device=device))

    torch.manual_seed(train_config.seed)
    model = TargetConditionedCircuitDenoiser(T=schedule.T, hidden=train_config.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    losses: list[float] = []

    iterator = range(1, train_config.num_steps + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Training target-conditioned circuit target", dynamic_ncols=True)

    for _ in iterator:
        rows = torch.randint(
            low=0,
            high=solution_stacks.shape[0],
            size=(train_config.batch_size,),
            device=device,
        )
        q0_stack = solution_stacks[rows]
        features = target_features[rows]
        t_idx = torch.randint(1, schedule.T + 1, (train_config.batch_size,), device=device)

        with torch.no_grad():
            qt_stack, eps_target = circuit_forward_heat_target(
                q0_stack,
                t_idx,
                schedule=schedule,
                n_terms=train_config.n_terms,
            )

        eps_pred = model(qt_stack, t_idx, features)
        loss = F.mse_loss(eps_pred, eps_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_value = loss.item()
        losses.append(loss_value)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_value:.5f}"})

    return model, losses


def train_target_conditioned_circuit_heat_kernel_model_synthetic(
    train_config: CircuitTrainConfig | None = None,
    schedule: DiffusionSchedule | None = None,
    data_config: DataConfig | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    entangler: str = "cz",
) -> tuple[TargetConditionedCircuitDenoiser, list[float]]:
    train_config = train_config or CircuitTrainConfig()
    schedule = schedule or DiffusionSchedule()
    data_config = data_config or DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    device = torch.device(device) if device is not None else get_default_device()

    torch.manual_seed(train_config.seed)
    centers = centers_for_config(data_config, device=device)
    center_names = center_names_for_config(data_config)
    entangler_unitary = two_qubit_gate(entangler, device=device)

    model = TargetConditionedCircuitDenoiser(T=schedule.T, hidden=train_config.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    losses: list[float] = []

    iterator = range(1, train_config.num_steps + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Training synthetic target-conditioned circuit target", dynamic_ncols=True)

    for _ in iterator:
        q0_stack, _ = sample_near_clifford_circuit_stacks(
            train_config.batch_size,
            centers=centers,
            center_names=center_names,
            sigma=data_config.sigma_data,
        )
        with torch.no_grad():
            target_unitaries = _compose_two_entangler_stack_units(
                quaternion_to_unitary(q0_stack),
                entangler_unitary,
            )
            features = target_unitary_features(target_unitaries)

        t_idx = torch.randint(1, schedule.T + 1, (train_config.batch_size,), device=device)

        with torch.no_grad():
            qt_stack, eps_target = circuit_forward_heat_target(
                q0_stack,
                t_idx,
                schedule=schedule,
                n_terms=train_config.n_terms,
            )

        eps_pred = model(qt_stack, t_idx, features)
        loss = F.mse_loss(eps_pred, eps_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_value = loss.item()
        losses.append(loss_value)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_value:.5f}"})

    return model, losses


def run_solution_stack_circuit_experiment(
    solution_stacks: torch.Tensor,
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> CircuitExperimentResult:
    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    device = torch.device(device) if device is not None else get_default_device()

    model, losses = train_circuit_heat_kernel_model_on_stacks(
        solution_stacks,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
    )
    with torch.no_grad():
        generated_deterministic = sample_circuit_reverse(
            model,
            config.schedule,
            n_samples=config.sample_count,
            eta=config.deterministic_eta,
            device=device,
        )
        generated_stochastic = sample_circuit_reverse(
            model,
            config.schedule,
            n_samples=config.sample_count,
            eta=config.eta,
            device=device,
        )
    return CircuitExperimentResult(
        config=config,
        model=model,
        losses=losses,
        generated_deterministic=generated_deterministic,
        generated_stochastic=generated_stochastic,
    )


def run_target_conditioned_solution_stack_circuit_experiment(
    solution_dataset: SolutionStackDatasetResult,
    eval_target_unitaries: torch.Tensor,
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> TargetConditionedCircuitExperimentResult:
    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if eval_target_unitaries.ndim != 3 or eval_target_unitaries.shape[1:] != (4, 4):
        raise ValueError("eval_target_unitaries must have shape (n_targets, 4, 4)")
    device = torch.device(device) if device is not None else get_default_device()
    train_target_unitaries = torch.stack([target.unitary for target in solution_dataset.targets])

    model, losses = train_target_conditioned_circuit_heat_kernel_model(
        solution_dataset.stacks,
        train_target_unitaries,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
    )
    with torch.no_grad():
        generated_deterministic_by_target = sample_target_conditioned_circuit_reverse(
            model,
            config.schedule,
            target_unitaries=eval_target_unitaries,
            n_samples_per_target=config.sample_count,
            eta=config.deterministic_eta,
            device=device,
        )
        generated_stochastic_by_target = sample_target_conditioned_circuit_reverse(
            model,
            config.schedule,
            target_unitaries=eval_target_unitaries,
            n_samples_per_target=config.sample_count,
            eta=config.eta,
            device=device,
        )
    return TargetConditionedCircuitExperimentResult(
        config=config,
        model=model,
        losses=losses,
        generated_deterministic_by_target=generated_deterministic_by_target,
        generated_stochastic_by_target=generated_stochastic_by_target,
    )


def run_target_conditioned_synthetic_circuit_experiment(
    eval_target_unitaries: torch.Tensor,
    config: CircuitExperimentConfig | str,
    data_config: DataConfig | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    entangler: str = "cz",
) -> TargetConditionedCircuitExperimentResult:
    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if eval_target_unitaries.ndim != 3 or eval_target_unitaries.shape[1:] != (4, 4):
        raise ValueError("eval_target_unitaries must have shape (n_targets, 4, 4)")
    device = torch.device(device) if device is not None else get_default_device()

    model, losses = train_target_conditioned_circuit_heat_kernel_model_synthetic(
        train_config=config.train,
        schedule=config.schedule,
        data_config=data_config or config.data,
        device=device,
        show_progress=show_progress,
        entangler=entangler,
    )
    with torch.no_grad():
        generated_deterministic_by_target = sample_target_conditioned_circuit_reverse(
            model,
            config.schedule,
            target_unitaries=eval_target_unitaries,
            n_samples_per_target=config.sample_count,
            eta=config.deterministic_eta,
            device=device,
        )
        generated_stochastic_by_target = sample_target_conditioned_circuit_reverse(
            model,
            config.schedule,
            target_unitaries=eval_target_unitaries,
            n_samples_per_target=config.sample_count,
            eta=config.eta,
            device=device,
        )
    return TargetConditionedCircuitExperimentResult(
        config=config,
        model=model,
        losses=losses,
        generated_deterministic_by_target=generated_deterministic_by_target,
        generated_stochastic_by_target=generated_stochastic_by_target,
    )


def run_target_conditioned_overfit_diagnostic(
    config: CircuitExperimentConfig | str,
    n_targets: int = 4,
    perturb_scale: float = 0.12,
    data_config: DataConfig | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    entangler: str = "cz",
    seed: int = 0,
    top_k: int = 5,
) -> TargetConditionedOverfitResult:
    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if n_targets <= 0:
        raise ValueError("n_targets must be positive")
    device = torch.device(device) if device is not None else get_default_device()
    data_config = data_config or config.data
    centers = centers_for_config(data_config, device=device)
    center_names = center_names_for_config(data_config)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    solution_stacks, _ = sample_near_clifford_circuit_stacks(
        n_targets,
        centers=centers,
        center_names=center_names,
        sigma=perturb_scale,
        generator=generator,
    )
    entangler_unitary = two_qubit_gate(entangler, device=device)
    target_unitaries = _compose_two_entangler_stack_units(
        quaternion_to_unitary(solution_stacks),
        entangler_unitary,
    )

    model, losses = train_target_conditioned_circuit_heat_kernel_model(
        solution_stacks,
        target_unitaries,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
    )
    with torch.no_grad():
        generated_stochastic_by_target = sample_target_conditioned_circuit_reverse(
            model,
            config.schedule,
            target_unitaries=target_unitaries,
            n_samples_per_target=config.sample_count,
            eta=config.eta,
            device=device,
        )
    reports = [
        synthesize_unitary_from_circuit_stack_report(
            stacks,
            target_unitary=target_unitary,
            target_name=f"overfit-{i:02d}",
            entangler=entangler,
            top_k=top_k,
            name=f"overfit-{i:02d} target-conditioned diffusion",
            keep_fidelities=False,
        )
        for i, (stacks, target_unitary) in enumerate(zip(generated_stochastic_by_target, target_unitaries))
    ]
    return TargetConditionedOverfitResult(
        target_unitaries=target_unitaries,
        solution_stacks=solution_stacks,
        reports=reports,
        losses=losses,
        generated_stochastic_by_target=generated_stochastic_by_target,
    )


@torch.no_grad()
def sample_circuit_reverse(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    n_samples: int = 5000,
    eta: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    n_slots = getattr(model, "n_slots", 6)
    betas, _, sigmas = schedule.tensors(device)
    q_stack = sample_haar(n_samples * n_slots, device=device).reshape(n_samples, n_slots, 4)

    for s in reversed(range(schedule.T)):
        t_idx = torch.full((n_samples,), s + 1, device=device, dtype=torch.long)
        eps_pred = model(q_stack, t_idx)

        beta = betas[s]
        sigma = sigmas[s]
        drift = -(beta / sigma.clamp_min(1e-8)) * eps_pred

        if s > 0 and eta > 0:
            noise = eta * torch.sqrt(beta) * torch.randn(n_samples, n_slots, 3, device=device)
        else:
            noise = torch.zeros_like(drift)

        q_stack = q_mul(q_stack, q_exp(drift + noise))
        q_stack = q_normalize(q_stack)

    return q_stack


@torch.no_grad()
def sample_target_conditioned_circuit_reverse(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    target_unitaries: torch.Tensor,
    n_samples_per_target: int = 1000,
    eta: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if target_unitaries.ndim != 3 or target_unitaries.shape[1:] != (4, 4):
        raise ValueError("target_unitaries must have shape (n_targets, 4, 4)")
    if n_samples_per_target <= 0:
        raise ValueError("n_samples_per_target must be positive")

    device = torch.device(device) if device is not None else next(model.parameters()).device
    target_unitaries = target_unitaries.to(device=device)
    n_targets = target_unitaries.shape[0]
    n_slots = getattr(model, "n_slots", 6)
    n_total = n_targets * n_samples_per_target
    features = target_unitary_features(target_unitaries)
    features = features[:, None, :].expand(n_targets, n_samples_per_target, features.shape[-1])
    features = features.reshape(n_total, features.shape[-1])

    betas, _, sigmas = schedule.tensors(device)
    q_stack = sample_haar(n_total * n_slots, device=device).reshape(n_total, n_slots, 4)

    for s in reversed(range(schedule.T)):
        t_idx = torch.full((n_total,), s + 1, device=device, dtype=torch.long)
        eps_pred = model(q_stack, t_idx, features)

        beta = betas[s]
        sigma = sigmas[s]
        drift = -(beta / sigma.clamp_min(1e-8)) * eps_pred

        if s > 0 and eta > 0:
            noise = eta * torch.sqrt(beta) * torch.randn(n_total, n_slots, 3, device=device)
        else:
            noise = torch.zeros_like(drift)

        q_stack = q_mul(q_stack, q_exp(drift + noise))
        q_stack = q_normalize(q_stack)

    return q_stack.reshape(n_targets, n_samples_per_target, n_slots, 4)


def run_circuit_experiment(
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> CircuitExperimentResult:
    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    device = torch.device(device) if device is not None else get_default_device()

    model, losses = train_circuit_heat_kernel_model(
        train_config=config.train,
        schedule=config.schedule,
        data_config=config.data,
        device=device,
        show_progress=show_progress,
    )
    with torch.no_grad():
        generated_deterministic = sample_circuit_reverse(
            model,
            config.schedule,
            n_samples=config.sample_count,
            eta=config.deterministic_eta,
            device=device,
        )
        generated_stochastic = sample_circuit_reverse(
            model,
            config.schedule,
            n_samples=config.sample_count,
            eta=config.eta,
            device=device,
        )

    return CircuitExperimentResult(
        config=config,
        model=model,
        losses=losses,
        generated_deterministic=generated_deterministic,
        generated_stochastic=generated_stochastic,
    )


def generate_solution_stack_dataset(
    clifford_gates: torch.Tensor,
    clifford_labels: list[str],
    n_targets: int = 128,
    perturb_scale: float = 0.12,
    entangler: str = "cz",
    candidate_count: int = 256,
    refinement_steps: int = 100,
    refinement_lr: float = 0.05,
    fidelity_threshold: float = 0.995,
    seed: int = 0,
) -> SolutionStackDatasetResult:
    if n_targets <= 0:
        raise ValueError("n_targets must be positive")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not (0.0 <= fidelity_threshold <= 1.0):
        raise ValueError("fidelity_threshold must be between 0 and 1")

    targets = make_near_clifford_two_entangler_circuit_targets(
        clifford_gates,
        clifford_labels,
        n_targets=n_targets,
        perturb_scale=perturb_scale,
        entangler=entangler,
        seed=seed,
    )
    stacks = []
    fidelities = []
    kept_targets = []
    refinements = []
    for i, target in enumerate(targets):
        candidate_stack = _best_analytic_stack_for_target(
            target,
            clifford_gates=clifford_gates,
            clifford_labels=clifford_labels,
            perturb_scale=perturb_scale,
            candidate_count=candidate_count,
            seed=seed + 10_000 + i,
        )
        units = quaternion_to_unitary(candidate_stack)
        entangler_unitary = two_qubit_gate(target.entangler, device=clifford_gates.device)
        unitary = compose_two_entangler_local(
            units[0],
            units[1],
            entangler_unitary,
            units[2],
            units[3],
            units[4],
            units[5],
        )
        candidate = SynthesisCandidate(
            target=target.name,
            template="solution-stack-analytic-start",
            entangler=target.entangler,
            fidelity=unitary_fidelity(unitary, target.unitary),
            slot_indices=(0, 1, 2, 3, 4, 5),
            slot_labels=("analytic", "analytic", "analytic", "analytic", "analytic", "analytic"),
        )
        refined = refine_two_entangler_candidate(
            candidate_stack,
            candidate,
            target_unitary=target.unitary,
            entangler=target.entangler,
            num_steps=refinement_steps,
            lr=refinement_lr,
        )
        if refined.refined_fidelity >= fidelity_threshold:
            stacks.append(refined.refined_gates)
            fidelities.append(refined.refined_fidelity)
            kept_targets.append(target)
            refinements.append(refined)

    if not stacks:
        raise RuntimeError("No solution stacks met the fidelity threshold")

    return SolutionStackDatasetResult(
        stacks=torch.stack(stacks),
        fidelities=torch.tensor(fidelities, dtype=torch.float32, device=clifford_gates.device),
        targets=kept_targets,
        refinements=refinements,
    )


def synthesize_unitary_from_circuit_stack_report(
    circuit_stacks: torch.Tensor,
    target_unitary: torch.Tensor,
    target_name: str,
    entangler: str = "cz",
    top_k: int = 5,
    name: str | None = None,
    keep_fidelities: bool = True,
) -> SynthesisReport:
    if circuit_stacks.ndim != 3 or circuit_stacks.shape[1:] != (6, 4):
        raise ValueError("circuit_stacks must have shape (n, 6, 4)")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    target_name = target_name.lower()
    device = circuit_stacks.device
    units = quaternion_to_unitary(circuit_stacks)
    entangler_unitary = two_qubit_gate(entangler, device=device)
    target_unitary = target_unitary.to(device=device, dtype=torch.complex64)

    unitaries = _compose_two_entangler_stack_units(units, entangler_unitary)
    fidelities = unitary_fidelity_batch(unitaries, target_unitary)
    values, rows = torch.topk(fidelities, k=min(top_k, fidelities.numel()))

    candidates = [
        SynthesisCandidate(
            target=target_name,
            template="joint-circuit-stack",
            entangler=entangler,
            fidelity=float(value),
            slot_indices=(int(row),) * 6,
            slot_labels=("joint", "joint", "joint", "joint", "joint", "joint"),
        )
        for value, row in zip(values.tolist(), rows.tolist())
    ]
    return SynthesisReport(
        name=name or f"{target_name} joint circuit diffusion",
        mode="joint-circuit",
        target=target_name,
        entangler=entangler,
        candidates=candidates,
        fidelities=tuple(fidelities.tolist()) if keep_fidelities else tuple(float(candidate.fidelity) for candidate in candidates),
    )


def run_joint_circuit_proposal_benchmark(
    benchmarks: list[NearCliffordCircuitBenchmark],
    circuit_stacks: torch.Tensor,
    top_k: int = 5,
    keep_fidelities: bool = True,
) -> list[SynthesisReport]:
    if not benchmarks:
        raise ValueError("run_joint_circuit_proposal_benchmark needs at least one benchmark")
    return [
        synthesize_unitary_from_circuit_stack_report(
            circuit_stacks,
            target_unitary=benchmark.target.unitary,
            target_name=benchmark.target.name,
            entangler=benchmark.target.entangler,
            top_k=top_k,
            name=f"{benchmark.target.name} joint circuit diffusion",
            keep_fidelities=keep_fidelities,
        )
        for benchmark in benchmarks
    ]


def run_target_conditioned_circuit_proposal_benchmark(
    benchmarks: list[NearCliffordCircuitBenchmark],
    circuit_stacks_by_target: torch.Tensor,
    top_k: int = 5,
    keep_fidelities: bool = True,
) -> list[SynthesisReport]:
    if not benchmarks:
        raise ValueError("run_target_conditioned_circuit_proposal_benchmark needs at least one benchmark")
    if circuit_stacks_by_target.ndim != 4 or circuit_stacks_by_target.shape[2:] != (6, 4):
        raise ValueError("circuit_stacks_by_target must have shape (n_targets, n_samples, 6, 4)")
    if circuit_stacks_by_target.shape[0] != len(benchmarks):
        raise ValueError("circuit_stacks_by_target must have one batch per benchmark")

    reports = []
    for benchmark, stacks in zip(benchmarks, circuit_stacks_by_target):
        reports.append(
            synthesize_unitary_from_circuit_stack_report(
                stacks,
                target_unitary=benchmark.target.unitary,
                target_name=benchmark.target.name,
                entangler=benchmark.target.entangler,
                top_k=top_k,
                name=f"{benchmark.target.name} target-conditioned circuit diffusion",
                keep_fidelities=keep_fidelities,
            )
        )
    return reports


def summarize_joint_circuit_comparison(
    benchmarks: list[NearCliffordCircuitBenchmark],
    joint_reports: list[SynthesisReport],
) -> list[HiddenShallowCircuitAggregate]:
    if len(benchmarks) != len(joint_reports):
        raise ValueError("benchmarks and joint_reports must have the same length")
    rows = summarize_near_clifford_two_entangler_benchmark(benchmarks)
    rows.append(_aggregate_reports("joint circuit diffusion", joint_reports))
    return rows


def summarize_solution_stack_circuit_comparison(
    benchmarks: list[NearCliffordCircuitBenchmark],
    random_joint_reports: list[SynthesisReport],
    solution_joint_reports: list[SynthesisReport],
) -> list[HiddenShallowCircuitAggregate]:
    rows = summarize_joint_circuit_comparison(benchmarks, random_joint_reports)
    rows.append(_aggregate_reports("solution-stack diffusion", solution_joint_reports))
    return rows


def summarize_target_conditioned_circuit_comparison(
    benchmarks: list[NearCliffordCircuitBenchmark],
    random_joint_reports: list[SynthesisReport],
    conditioned_reports: list[SynthesisReport],
    solution_joint_reports: list[SynthesisReport] | None = None,
) -> list[HiddenShallowCircuitAggregate]:
    rows = summarize_joint_circuit_comparison(benchmarks, random_joint_reports)
    if solution_joint_reports is not None:
        rows.append(_aggregate_reports("solution-stack diffusion", solution_joint_reports))
    rows.append(_aggregate_reports("target-conditioned diffusion", conditioned_reports))
    return rows


def print_solution_stack_dataset_summary(dataset: SolutionStackDatasetResult) -> None:
    header = "n stacks   mean fidelity   min fidelity   max fidelity"
    print(header)
    print("-" * len(header))
    print(
        f"{dataset.stacks.shape[0]:<8}   {dataset.fidelities.mean().item():>11.4f}   "
        f"{dataset.fidelities.min().item():>10.4f}   {dataset.fidelities.max().item():>10.4f}"
    )


def print_solution_stack_circuit_comparison_summary(
    benchmarks: list[NearCliffordCircuitBenchmark],
    random_joint_reports: list[SynthesisReport],
    solution_joint_reports: list[SynthesisReport],
) -> None:
    header = "mode                      n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_solution_stack_circuit_comparison(benchmarks, random_joint_reports, solution_joint_reports):
        print(
            f"{item.mode:<25} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_target_conditioned_circuit_comparison_summary(
    benchmarks: list[NearCliffordCircuitBenchmark],
    random_joint_reports: list[SynthesisReport],
    conditioned_reports: list[SynthesisReport],
    solution_joint_reports: list[SynthesisReport] | None = None,
) -> None:
    header = "mode                           n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_target_conditioned_circuit_comparison(
        benchmarks,
        random_joint_reports,
        conditioned_reports,
        solution_joint_reports=solution_joint_reports,
    ):
        print(
            f"{item.mode:<30} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_target_conditioned_overfit_summary(result: TargetConditionedOverfitResult) -> None:
    header = "target       best fidelity"
    print(header)
    print("-" * len(header))
    for report in result.reports:
        print(f"{report.target:<12} {report.candidates[0].fidelity:>12.4f}")
    aggregate = _aggregate_reports("target-conditioned overfit", result.reports)
    print()
    header = "mode                         n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    print(
        f"{aggregate.mode:<28} {aggregate.n_targets:<3} "
        f"{aggregate.mean_best:>9.4f}   {aggregate.median_best:>6.4f}   "
        f"{aggregate.min_best:>6.4f}   {aggregate.max_best:>6.4f}   "
        f"{aggregate.success_95:>6.1%}   {aggregate.success_98:>6.1%}   {aggregate.success_99:>6.1%}"
    )


def print_joint_circuit_comparison_summary(
    benchmarks: list[NearCliffordCircuitBenchmark],
    joint_reports: list[SynthesisReport],
) -> None:
    header = "mode                      n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_joint_circuit_comparison(benchmarks, joint_reports):
        print(
            f"{item.mode:<25} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def plot_joint_circuit_comparison(
    benchmarks: list[NearCliffordCircuitBenchmark],
    joint_reports: list[SynthesisReport],
) -> None:
    if len(benchmarks) != len(joint_reports):
        raise ValueError("benchmarks and joint_reports must have the same length")

    values = [
        _best_values([item.clifford_report for item in benchmarks]),
        _best_values([item.analytic_report for item in benchmarks]),
        _best_values([item.generated_report for item in benchmarks]),
        _best_values([item.haar_report for item in benchmarks]),
        _best_values(joint_reports),
    ]
    labels = ["Clifford", "analytic", "independent diffusion", "Haar", "joint diffusion"]

    plt.figure(figsize=(10, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Near-Clifford proposals: independent gates vs joint circuit diffusion")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_solution_stack_circuit_comparison(
    benchmarks: list[NearCliffordCircuitBenchmark],
    random_joint_reports: list[SynthesisReport],
    solution_joint_reports: list[SynthesisReport],
) -> None:
    values = [
        _best_values([item.clifford_report for item in benchmarks]),
        _best_values([item.analytic_report for item in benchmarks]),
        _best_values([item.generated_report for item in benchmarks]),
        _best_values([item.haar_report for item in benchmarks]),
        _best_values(random_joint_reports),
        _best_values(solution_joint_reports),
    ]
    labels = ["Clifford", "analytic", "independent", "Haar", "joint random", "joint solution"]

    plt.figure(figsize=(11, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Near-Clifford proposals with solution-stack diffusion")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_target_conditioned_circuit_comparison(
    benchmarks: list[NearCliffordCircuitBenchmark],
    random_joint_reports: list[SynthesisReport],
    conditioned_reports: list[SynthesisReport],
    solution_joint_reports: list[SynthesisReport] | None = None,
) -> None:
    values = [
        _best_values([item.clifford_report for item in benchmarks]),
        _best_values([item.analytic_report for item in benchmarks]),
        _best_values([item.generated_report for item in benchmarks]),
        _best_values([item.haar_report for item in benchmarks]),
        _best_values(random_joint_reports),
        _best_values(conditioned_reports),
    ]
    labels = ["Clifford", "analytic", "independent", "Haar", "joint random", "conditioned"]
    if solution_joint_reports is not None:
        values.insert(-1, _best_values(solution_joint_reports))
        labels.insert(-1, "joint solution")

    plt.figure(figsize=(12, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Target-conditioned circuit diffusion proposals")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_target_conditioned_overfit(result: TargetConditionedOverfitResult) -> None:
    values = _best_values(result.reports)
    plt.figure(figsize=(6, 4))
    plt.boxplot([values], labels=["overfit targets"], showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Target-conditioned tiny-set overfit diagnostic")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def _compose_two_entangler_stack_units(
    units: torch.Tensor,
    entangler_unitary: torch.Tensor,
) -> torch.Tensor:
    first = _batched_local_layer(units[:, 0], units[:, 1])
    middle = _batched_local_layer(units[:, 2], units[:, 3])
    second = _batched_local_layer(units[:, 4], units[:, 5])
    entanglers = entangler_unitary.expand(units.shape[0], 4, 4)
    return first @ entanglers @ middle @ entanglers @ second


def _best_analytic_stack_for_target(
    target: HiddenShallowCircuitTarget,
    clifford_gates: torch.Tensor,
    clifford_labels: list[str],
    perturb_scale: float,
    candidate_count: int,
    seed: int,
) -> torch.Tensor:
    flat_gates, _ = sample_near_clifford_gates(
        clifford_gates,
        clifford_labels,
        n_samples=candidate_count * 6,
        perturb_scale=perturb_scale,
        seed=seed,
    )
    stacks = flat_gates.reshape(candidate_count, 6, 4)
    units = quaternion_to_unitary(stacks)
    entangler_unitary = two_qubit_gate(target.entangler, device=clifford_gates.device)
    unitaries = _compose_two_entangler_stack_units(units, entangler_unitary)
    fidelities = unitary_fidelity_batch(unitaries, target.unitary)
    return stacks[int(torch.argmax(fidelities).item())]


def _batched_local_layer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nab,ncd->nacbd", a, b).reshape(a.shape[0], 4, 4)


def _best_values(reports: list[SynthesisReport]) -> torch.Tensor:
    return torch.tensor([report.candidates[0].fidelity for report in reports], dtype=torch.float32)


def _aggregate_reports(mode: str, reports: list[SynthesisReport]) -> HiddenShallowCircuitAggregate:
    values = _best_values(reports)
    return HiddenShallowCircuitAggregate(
        mode=mode,
        n_targets=values.numel(),
        mean_best=values.mean().item(),
        median_best=values.median().item(),
        min_best=values.min().item(),
        max_best=values.max().item(),
        success_95=(values >= 0.95).float().mean().item(),
        success_98=(values >= 0.98).float().mean().item(),
        success_99=(values >= 0.99).float().mean().item(),
    )
