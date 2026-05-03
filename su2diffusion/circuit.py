from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import DataConfig, center_names_for_config, centers_for_config
from .device import get_default_device
from .diffusion import DiffusionSchedule, brownian_forward_heat_target
from .model import CircuitDenoiser
from .quaternion import q_exp, q_mul, q_normalize, sample_haar
from .synthesis import (
    HiddenShallowCircuitAggregate,
    NearCliffordCircuitBenchmark,
    SynthesisCandidate,
    SynthesisReport,
    compose_two_entangler_local,
    quaternion_to_unitary,
    summarize_near_clifford_two_entangler_benchmark,
    two_qubit_gate,
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


def summarize_joint_circuit_comparison(
    benchmarks: list[NearCliffordCircuitBenchmark],
    joint_reports: list[SynthesisReport],
) -> list[HiddenShallowCircuitAggregate]:
    if len(benchmarks) != len(joint_reports):
        raise ValueError("benchmarks and joint_reports must have the same length")
    rows = summarize_near_clifford_two_entangler_benchmark(benchmarks)
    rows.append(_aggregate_reports("joint circuit diffusion", joint_reports))
    return rows


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


def _compose_two_entangler_stack_units(
    units: torch.Tensor,
    entangler_unitary: torch.Tensor,
) -> torch.Tensor:
    first = _batched_local_layer(units[:, 0], units[:, 1])
    middle = _batched_local_layer(units[:, 2], units[:, 3])
    second = _batched_local_layer(units[:, 4], units[:, 5])
    entanglers = entangler_unitary.expand(units.shape[0], 4, 4)
    return first @ entanglers @ middle @ entanglers @ second


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
