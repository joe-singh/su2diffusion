from dataclasses import dataclass
import re

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quaternion import q_normalize, sample_haar
from .synthesis import (
    HiddenShallowCircuitAggregate,
    RefinementResult,
    SynthesisCandidate,
    SynthesisReport,
    compose_two_entangler_local,
    quaternion_to_unitary,
    refine_two_entangler_candidate,
    sample_near_clifford_gates,
    synthesize_unitary_two_entangler_random_report,
    two_qubit_gate,
    unitary_fidelity,
)


@dataclass(frozen=True)
class HamiltonianTerm:
    pauli: str
    coefficient: float


@dataclass(frozen=True)
class HamiltonianTarget:
    name: str
    terms: tuple[HamiltonianTerm, ...]
    time: float
    hamiltonian: torch.Tensor
    unitary: torch.Tensor


@dataclass(frozen=True)
class HamiltonianSynthesisBenchmark:
    target: HamiltonianTarget
    clifford_report: SynthesisReport
    analytic_report: SynthesisReport
    generated_report: SynthesisReport
    haar_report: SynthesisReport


@dataclass(frozen=True)
class HamiltonianSuiteResult:
    benchmarks: list[HamiltonianSynthesisBenchmark]


@dataclass(frozen=True)
class HamiltonianSolutionDataset:
    targets: list[HamiltonianTarget]
    benchmarks: list[HamiltonianSynthesisBenchmark]
    refinements: list[RefinementResult]
    stacks: torch.Tensor
    initial_fidelities: torch.Tensor
    refined_fidelities: torch.Tensor


@dataclass(frozen=True)
class HamiltonianSupervisedTrainConfig:
    hidden: int = 256
    num_steps: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


@dataclass
class HamiltonianSupervisedResult:
    model: torch.nn.Module
    losses: list[float]
    predicted_stacks: torch.Tensor
    raw_fidelities: torch.Tensor
    refined_results: list[RefinementResult] | None = None


@dataclass
class HamiltonianSupervisedSplitResult:
    train: HamiltonianSupervisedResult
    heldout: HamiltonianSupervisedResult


class HamiltonianStackPredictor(nn.Module):
    def __init__(self, input_dim: int = 33, hidden: int = 256, n_slots: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.n_slots = n_slots
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_slots * 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"Expected features with shape (batch, {self.input_dim})")
        return q_normalize(self.net(features).reshape(features.shape[0], self.n_slots, 4))


def pauli_matrix(name: str, device: torch.device | str | None = None) -> torch.Tensor:
    name = name.upper()
    if name == "I":
        matrix = [[1, 0], [0, 1]]
    elif name == "X":
        matrix = [[0, 1], [1, 0]]
    elif name == "Y":
        matrix = [[0, -1j], [1j, 0]]
    elif name == "Z":
        matrix = [[1, 0], [0, -1]]
    else:
        raise ValueError(f"Unknown Pauli matrix {name!r}")
    return torch.tensor(matrix, dtype=torch.complex64, device=device)


def parse_pauli_string(pauli: str, n_qubits: int = 2) -> tuple[str, ...]:
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    compact = pauli.replace(" ", "").upper()
    if len(compact) == n_qubits and all(item in "IXYZ" for item in compact):
        return tuple(compact)

    factors = ["I"] * n_qubits
    matches = re.findall(r"([IXYZ])\s*([0-9]+)", pauli.upper())
    if not matches:
        raise ValueError(f"Could not parse Pauli string {pauli!r}")

    consumed = "".join(f"{gate}{index}" for gate, index in matches)
    if consumed != compact:
        raise ValueError(f"Could not parse Pauli string {pauli!r}")

    for gate, index_text in matches:
        index = int(index_text)
        if index < 0 or index >= n_qubits:
            raise ValueError(f"Qubit index {index} is outside n_qubits={n_qubits}")
        if factors[index] != "I":
            raise ValueError(f"Qubit index {index} appears more than once in {pauli!r}")
        factors[index] = gate
    return tuple(factors)


def pauli_string_matrix(
    pauli: str,
    n_qubits: int = 2,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    factors = [pauli_matrix(item, device=device) for item in parse_pauli_string(pauli, n_qubits=n_qubits)]
    result = factors[0]
    for factor in factors[1:]:
        result = torch.kron(result, factor)
    return result


def _coerce_term(term: HamiltonianTerm | tuple[str, float] | tuple[float, str]) -> HamiltonianTerm:
    if isinstance(term, HamiltonianTerm):
        return term
    if len(term) != 2:
        raise ValueError("Hamiltonian terms must be (pauli, coefficient) pairs")
    first, second = term
    if isinstance(first, str):
        return HamiltonianTerm(pauli=first, coefficient=float(second))
    if isinstance(second, str):
        return HamiltonianTerm(pauli=second, coefficient=float(first))
    raise ValueError("Hamiltonian terms must include one Pauli string and one coefficient")


def hamiltonian_from_terms(
    terms: list[HamiltonianTerm | tuple[str, float] | tuple[float, str]] | tuple[HamiltonianTerm | tuple[str, float] | tuple[float, str], ...],
    n_qubits: int = 2,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not terms:
        raise ValueError("terms must contain at least one Hamiltonian term")
    dim = 2**n_qubits
    hamiltonian = torch.zeros(dim, dim, dtype=torch.complex64, device=device)
    for raw_term in terms:
        term = _coerce_term(raw_term)
        hamiltonian = hamiltonian + term.coefficient * pauli_string_matrix(
            term.pauli,
            n_qubits=n_qubits,
            device=device,
        )
    return hamiltonian


def unitary_from_hamiltonian(hamiltonian: torch.Tensor, time: float = 1.0) -> torch.Tensor:
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be a square matrix")
    return torch.linalg.matrix_exp(-1j * float(time) * hamiltonian.to(dtype=torch.complex64))


def make_hamiltonian_target(
    terms: list[HamiltonianTerm | tuple[str, float] | tuple[float, str]] | tuple[HamiltonianTerm | tuple[str, float] | tuple[float, str], ...],
    time: float = 1.0,
    name: str = "hamiltonian",
    n_qubits: int = 2,
    device: torch.device | str | None = None,
) -> HamiltonianTarget:
    coerced_terms = tuple(_coerce_term(term) for term in terms)
    hamiltonian = hamiltonian_from_terms(coerced_terms, n_qubits=n_qubits, device=device)
    unitary = unitary_from_hamiltonian(hamiltonian, time=time)
    return HamiltonianTarget(
        name=name,
        terms=coerced_terms,
        time=float(time),
        hamiltonian=hamiltonian,
        unitary=unitary,
    )


def make_random_pauli_hamiltonian_targets(
    n_targets: int = 12,
    terms: tuple[str, ...] = ("XI", "IZ", "XX", "ZZ"),
    coefficient_scale: float = 0.35,
    time: float = 0.8,
    name_prefix: str = "pauli",
    seed: int = 0,
    device: torch.device | str | None = None,
) -> list[HamiltonianTarget]:
    if n_targets <= 0:
        raise ValueError("n_targets must be positive")
    if not terms:
        raise ValueError("terms must contain at least one Pauli string")
    if coefficient_scale <= 0:
        raise ValueError("coefficient_scale must be positive")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    coefficients = coefficient_scale * torch.randn(n_targets, len(terms), device=device, generator=generator)
    targets = []
    for i in range(n_targets):
        target_terms = tuple(
            HamiltonianTerm(pauli=pauli, coefficient=float(coefficient))
            for pauli, coefficient in zip(terms, coefficients[i].tolist())
        )
        targets.append(
            make_hamiltonian_target(
                target_terms,
                time=time,
                name=f"{name_prefix}-{i:02d}",
                device=device,
            )
        )
    return targets


def hamiltonian_target_features(targets: list[HamiltonianTarget]) -> torch.Tensor:
    if not targets:
        raise ValueError("targets must contain at least one Hamiltonian target")
    hamiltonians = torch.stack([target.hamiltonian for target in targets]).to(dtype=torch.complex64)
    times = torch.tensor(
        [target.time for target in targets],
        dtype=torch.float32,
        device=hamiltonians.device,
    )[:, None]
    return torch.cat(
        [
            hamiltonians.real.reshape(len(targets), -1),
            hamiltonians.imag.reshape(len(targets), -1),
            times,
        ],
        dim=-1,
    )


def run_hamiltonian_two_entangler_benchmark(
    target: HamiltonianTarget,
    clifford_gates: torch.Tensor,
    clifford_labels: list[str],
    generated_gates: torch.Tensor,
    generated_labels: list[str],
    perturb_scale: float = 0.12,
    entangler: str = "cz",
    n_random_candidates: int = 200_000,
    n_analytic_gates: int = 1024,
    n_haar_gates: int = 1024,
    top_k: int = 5,
    seed: int = 0,
    keep_fidelities: bool = True,
) -> HamiltonianSynthesisBenchmark:
    if n_analytic_gates <= 0:
        raise ValueError("n_analytic_gates must be positive")
    if n_haar_gates <= 0:
        raise ValueError("n_haar_gates must be positive")

    device = clifford_gates.device
    target_unitary = target.unitary.to(device=device)
    analytic_gates, analytic_labels = sample_near_clifford_gates(
        clifford_gates,
        clifford_labels,
        n_samples=n_analytic_gates,
        perturb_scale=perturb_scale,
        seed=seed + 25_000,
    )
    haar_generator = torch.Generator(device=device)
    haar_generator.manual_seed(seed + 30_000)
    haar_gates = sample_haar(n_haar_gates, device=device, generator=haar_generator)
    haar_labels = ["Haar"] * n_haar_gates

    clifford_report = synthesize_unitary_two_entangler_random_report(
        clifford_gates,
        target_unitary=target_unitary,
        target_name=target.name,
        entangler=entangler,
        n_candidates=n_random_candidates,
        top_k=top_k,
        local_labels=clifford_labels,
        seed=seed + 10_000,
        name=f"{target.name} Clifford random",
        keep_fidelities=keep_fidelities,
    )
    analytic_report = synthesize_unitary_two_entangler_random_report(
        analytic_gates,
        target_unitary=target_unitary,
        target_name=target.name,
        entangler=entangler,
        n_candidates=n_random_candidates,
        top_k=top_k,
        local_labels=analytic_labels,
        seed=seed + 15_000,
        name=f"{target.name} analytic near-Clifford random",
        keep_fidelities=keep_fidelities,
    )
    generated_report = synthesize_unitary_two_entangler_random_report(
        generated_gates,
        target_unitary=target_unitary,
        target_name=target.name,
        entangler=entangler,
        n_candidates=n_random_candidates,
        top_k=top_k,
        local_labels=generated_labels,
        seed=seed + 20_000,
        name=f"{target.name} generated random",
        keep_fidelities=keep_fidelities,
    )
    haar_report = synthesize_unitary_two_entangler_random_report(
        haar_gates,
        target_unitary=target_unitary,
        target_name=target.name,
        entangler=entangler,
        n_candidates=n_random_candidates,
        top_k=top_k,
        local_labels=haar_labels,
        seed=seed + 40_000,
        name=f"{target.name} Haar random",
        keep_fidelities=keep_fidelities,
    )
    return HamiltonianSynthesisBenchmark(
        target=target,
        clifford_report=clifford_report,
        analytic_report=analytic_report,
        generated_report=generated_report,
        haar_report=haar_report,
    )


def run_hamiltonian_suite_benchmark(
    targets: list[HamiltonianTarget],
    clifford_gates: torch.Tensor,
    clifford_labels: list[str],
    generated_gates: torch.Tensor,
    generated_labels: list[str],
    perturb_scale: float = 0.12,
    entangler: str = "cz",
    n_random_candidates: int = 100_000,
    n_analytic_gates: int = 1024,
    n_haar_gates: int = 1024,
    top_k: int = 5,
    seed: int = 0,
    keep_fidelities: bool = False,
) -> HamiltonianSuiteResult:
    if not targets:
        raise ValueError("targets must contain at least one Hamiltonian target")
    benchmarks = [
        run_hamiltonian_two_entangler_benchmark(
            target,
            clifford_gates=clifford_gates,
            clifford_labels=clifford_labels,
            generated_gates=generated_gates,
            generated_labels=generated_labels,
            perturb_scale=perturb_scale,
            entangler=entangler,
            n_random_candidates=n_random_candidates,
            n_analytic_gates=n_analytic_gates,
            n_haar_gates=n_haar_gates,
            top_k=top_k,
            seed=seed + i,
            keep_fidelities=keep_fidelities,
        )
        for i, target in enumerate(targets)
    ]
    return HamiltonianSuiteResult(benchmarks=benchmarks)


def generate_hamiltonian_solution_dataset(
    targets: list[HamiltonianTarget],
    clifford_gates: torch.Tensor,
    clifford_labels: list[str],
    generated_gates: torch.Tensor,
    generated_labels: list[str],
    perturb_scale: float = 0.12,
    entangler: str = "cz",
    n_random_candidates: int = 100_000,
    n_analytic_gates: int = 1024,
    n_haar_gates: int = 1024,
    top_k: int = 5,
    seed: int = 0,
    refinement_steps: int = 200,
    refinement_lr: float = 0.05,
    fidelity_threshold: float = 0.0,
) -> HamiltonianSolutionDataset:
    if not targets:
        raise ValueError("targets must contain at least one Hamiltonian target")
    if refinement_steps <= 0:
        raise ValueError("refinement_steps must be positive")
    if refinement_lr <= 0:
        raise ValueError("refinement_lr must be positive")
    if not (0.0 <= fidelity_threshold <= 1.0):
        raise ValueError("fidelity_threshold must be between 0 and 1")

    suite = run_hamiltonian_suite_benchmark(
        targets,
        clifford_gates=clifford_gates,
        clifford_labels=clifford_labels,
        generated_gates=generated_gates,
        generated_labels=generated_labels,
        perturb_scale=perturb_scale,
        entangler=entangler,
        n_random_candidates=n_random_candidates,
        n_analytic_gates=n_analytic_gates,
        n_haar_gates=n_haar_gates,
        top_k=top_k,
        seed=seed,
        keep_fidelities=False,
    )

    kept_targets = []
    kept_benchmarks = []
    refinements = []
    for benchmark in suite.benchmarks:
        candidate = benchmark.generated_report.candidates[0]
        refinement = refine_two_entangler_candidate(
            generated_gates,
            candidate,
            target_unitary=benchmark.target.unitary,
            entangler=entangler,
            num_steps=refinement_steps,
            lr=refinement_lr,
        )
        if refinement.refined_fidelity >= fidelity_threshold:
            kept_targets.append(benchmark.target)
            kept_benchmarks.append(benchmark)
            refinements.append(refinement)

    if not refinements:
        raise RuntimeError("No Hamiltonian solution stacks met the fidelity threshold")

    device = generated_gates.device
    return HamiltonianSolutionDataset(
        targets=kept_targets,
        benchmarks=kept_benchmarks,
        refinements=refinements,
        stacks=torch.stack([item.refined_gates for item in refinements]),
        initial_fidelities=torch.tensor([item.initial_fidelity for item in refinements], dtype=torch.float32, device=device),
        refined_fidelities=torch.tensor([item.refined_fidelity for item in refinements], dtype=torch.float32, device=device),
    )


def _stack_unitary(q_stack: torch.Tensor, entangler: str = "cz") -> torch.Tensor:
    units = quaternion_to_unitary(q_stack)
    entangler_unitary = two_qubit_gate(entangler, device=q_stack.device)
    return compose_two_entangler_local(
        units[0],
        units[1],
        entangler_unitary,
        units[2],
        units[3],
        units[4],
        units[5],
    )


def _stack_fidelity(q_stack: torch.Tensor, target: HamiltonianTarget, entangler: str = "cz") -> float:
    return unitary_fidelity(_stack_unitary(q_stack, entangler=entangler), target.unitary)


def _aligned_stack_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direct = (predicted - target).square().mean(dim=(-1, -2))
    flipped = (predicted + target).square().mean(dim=(-1, -2))
    return torch.minimum(direct, flipped).mean()


def train_hamiltonian_stack_predictor(
    dataset: HamiltonianSolutionDataset,
    config: HamiltonianSupervisedTrainConfig | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> tuple[HamiltonianStackPredictor, list[float]]:
    if not dataset.targets:
        raise ValueError("dataset must contain at least one target")
    config = config or HamiltonianSupervisedTrainConfig()
    device = torch.device(device) if device is not None else dataset.stacks.device

    torch.manual_seed(config.seed)
    features = hamiltonian_target_features(dataset.targets).to(device=device)
    stacks = dataset.stacks.to(device=device)
    model = HamiltonianStackPredictor(
        input_dim=features.shape[1],
        hidden=config.hidden,
        n_slots=stacks.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    losses: list[float] = []
    iterator = range(1, config.num_steps + 1)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Training Hamiltonian stack predictor", dynamic_ncols=True)

    for _ in iterator:
        predicted = model(features)
        mse_loss = _aligned_stack_mse(predicted, stacks)
        norm_loss = (predicted.norm(dim=-1) - 1.0).square().mean()
        loss = mse_loss + 0.01 * norm_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_value:.5f}"})

    return model, losses


@torch.no_grad()
def predict_hamiltonian_stacks(
    model: HamiltonianStackPredictor,
    targets: list[HamiltonianTarget],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    device = torch.device(device) if device is not None else next(model.parameters()).device
    features = hamiltonian_target_features(targets).to(device=device)
    return model(features)


def evaluate_hamiltonian_stack_predictor(
    model: HamiltonianStackPredictor,
    targets: list[HamiltonianTarget],
    device: torch.device | str | None = None,
    entangler: str = "cz",
    refine: bool = False,
    refinement_steps: int = 100,
    refinement_lr: float = 0.05,
) -> HamiltonianSupervisedResult:
    if not targets:
        raise ValueError("targets must contain at least one Hamiltonian target")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    predicted_stacks = predict_hamiltonian_stacks(model, targets, device=device)
    raw_fidelities = torch.tensor(
        [
            _stack_fidelity(stack, target, entangler=entangler)
            for stack, target in zip(predicted_stacks, targets)
        ],
        dtype=torch.float32,
        device=device,
    )

    refined_results = None
    if refine:
        refined_results = []
        for stack, target, fidelity in zip(predicted_stacks, targets, raw_fidelities.tolist()):
            candidate = SynthesisCandidate(
                target=target.name,
                template="hamiltonian-supervised-stack",
                entangler=entangler,
                fidelity=fidelity,
                slot_indices=(0, 1, 2, 3, 4, 5),
                slot_labels=("predicted",) * 6,
            )
            refined_results.append(
                refine_two_entangler_candidate(
                    stack,
                    candidate,
                    target_unitary=target.unitary,
                    entangler=entangler,
                    num_steps=refinement_steps,
                    lr=refinement_lr,
                )
            )

    return HamiltonianSupervisedResult(
        model=model,
        losses=[],
        predicted_stacks=predicted_stacks.detach(),
        raw_fidelities=raw_fidelities,
        refined_results=refined_results,
    )


def run_hamiltonian_supervised_baseline(
    train_dataset: HamiltonianSolutionDataset,
    eval_targets: list[HamiltonianTarget] | None = None,
    config: HamiltonianSupervisedTrainConfig | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    entangler: str = "cz",
    refine: bool = True,
    refinement_steps: int = 100,
    refinement_lr: float = 0.05,
) -> HamiltonianSupervisedResult:
    model, losses = train_hamiltonian_stack_predictor(
        train_dataset,
        config=config,
        device=device,
        show_progress=show_progress,
    )
    result = evaluate_hamiltonian_stack_predictor(
        model,
        eval_targets or train_dataset.targets,
        device=device,
        entangler=entangler,
        refine=refine,
        refinement_steps=refinement_steps,
        refinement_lr=refinement_lr,
    )
    result.losses.extend(losses)
    return result


def run_hamiltonian_supervised_split_baseline(
    train_dataset: HamiltonianSolutionDataset,
    heldout_targets: list[HamiltonianTarget],
    config: HamiltonianSupervisedTrainConfig | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    entangler: str = "cz",
    refine: bool = True,
    refinement_steps: int = 100,
    refinement_lr: float = 0.05,
) -> HamiltonianSupervisedSplitResult:
    if not heldout_targets:
        raise ValueError("heldout_targets must contain at least one target")
    model, losses = train_hamiltonian_stack_predictor(
        train_dataset,
        config=config,
        device=device,
        show_progress=show_progress,
    )
    train_result = evaluate_hamiltonian_stack_predictor(
        model,
        train_dataset.targets,
        device=device,
        entangler=entangler,
        refine=refine,
        refinement_steps=refinement_steps,
        refinement_lr=refinement_lr,
    )
    train_result.losses.extend(losses)
    heldout_result = evaluate_hamiltonian_stack_predictor(
        model,
        heldout_targets,
        device=device,
        entangler=entangler,
        refine=refine,
        refinement_steps=refinement_steps,
        refinement_lr=refinement_lr,
    )
    heldout_result.losses.extend(losses)
    return HamiltonianSupervisedSplitResult(train=train_result, heldout=heldout_result)


def _best(report: SynthesisReport) -> float:
    if not report.candidates:
        raise ValueError("report has no candidates")
    return report.candidates[0].fidelity


def summarize_hamiltonian_two_entangler_benchmark(
    benchmark: HamiltonianSynthesisBenchmark,
) -> list[HiddenShallowCircuitAggregate]:
    reports = [
        ("Clifford random", benchmark.clifford_report),
        ("analytic near-Clifford", benchmark.analytic_report),
        ("generated random", benchmark.generated_report),
        ("Haar random", benchmark.haar_report),
    ]
    rows = []
    for mode, report in reports:
        best = _best(report)
        rows.append(
            HiddenShallowCircuitAggregate(
                mode=mode,
                n_targets=1,
                mean_best=best,
                median_best=best,
                min_best=best,
                max_best=best,
                success_95=float(best >= 0.95),
                success_98=float(best >= 0.98),
                success_99=float(best >= 0.99),
            )
        )
    return rows


def _aggregate_reports(mode: str, reports: list[SynthesisReport]) -> HiddenShallowCircuitAggregate:
    if not reports:
        raise ValueError("reports must contain at least one report")
    values = torch.tensor([_best(report) for report in reports], dtype=torch.float32)
    return HiddenShallowCircuitAggregate(
        mode=mode,
        n_targets=len(reports),
        mean_best=float(values.mean().item()),
        median_best=float(values.median().item()),
        min_best=float(values.min().item()),
        max_best=float(values.max().item()),
        success_95=float((values >= 0.95).float().mean().item()),
        success_98=float((values >= 0.98).float().mean().item()),
        success_99=float((values >= 0.99).float().mean().item()),
    )


def summarize_hamiltonian_suite(result: HamiltonianSuiteResult) -> list[HiddenShallowCircuitAggregate]:
    if not result.benchmarks:
        raise ValueError("result must contain at least one benchmark")
    return [
        _aggregate_reports("Clifford random", [item.clifford_report for item in result.benchmarks]),
        _aggregate_reports("analytic near-Clifford", [item.analytic_report for item in result.benchmarks]),
        _aggregate_reports("generated random", [item.generated_report for item in result.benchmarks]),
        _aggregate_reports("Haar random", [item.haar_report for item in result.benchmarks]),
    ]


def print_hamiltonian_target(target: HamiltonianTarget) -> None:
    print(f"target: {target.name}")
    print(f"time:   {target.time:g}")
    print("terms:")
    for term in target.terms:
        print(f"  {term.coefficient:+.4f} {term.pauli}")


def print_hamiltonian_two_entangler_benchmark(benchmark: HamiltonianSynthesisBenchmark) -> None:
    header = "mode                    best fidelity   best labels"
    print(header)
    print("-" * len(header))
    for mode, report in [
        ("Clifford random", benchmark.clifford_report),
        ("analytic near-Clifford", benchmark.analytic_report),
        ("generated random", benchmark.generated_report),
        ("Haar random", benchmark.haar_report),
    ]:
        labels = ", ".join(label if label is not None else "?" for label in report.candidates[0].slot_labels)
        print(f"{mode:<23} {_best(report):>12.4f}   {labels}")


def print_hamiltonian_two_entangler_summary(benchmark: HamiltonianSynthesisBenchmark) -> None:
    header = "mode                   n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_hamiltonian_two_entangler_benchmark(benchmark):
        print(
            f"{item.mode:<22} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_hamiltonian_suite(result: HamiltonianSuiteResult, max_rows: int | None = 6) -> None:
    header = "target      Clifford analytic generated Haar"
    print(header)
    print("-" * len(header))
    rows = result.benchmarks if max_rows is None else result.benchmarks[:max_rows]
    for item in rows:
        print(
            f"{item.target.name:<11} "
            f"{_best(item.clifford_report):>8.4f} "
            f"{_best(item.analytic_report):>8.4f} "
            f"{_best(item.generated_report):>9.4f} "
            f"{_best(item.haar_report):>6.4f}"
        )
    if max_rows is not None and len(result.benchmarks) > max_rows:
        print(f"... {len(result.benchmarks) - max_rows} more")


def print_hamiltonian_suite_summary(result: HamiltonianSuiteResult) -> None:
    header = "mode                   n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_hamiltonian_suite(result):
        print(
            f"{item.mode:<22} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_hamiltonian_solution_dataset(dataset: HamiltonianSolutionDataset, max_rows: int | None = 6) -> None:
    header = "target      before   after    gain"
    print(header)
    print("-" * len(header))
    rows = list(zip(dataset.targets, dataset.refinements))
    rows = rows if max_rows is None else rows[:max_rows]
    for target, refinement in rows:
        gain = refinement.refined_fidelity - refinement.initial_fidelity
        print(
            f"{target.name:<11} "
            f"{refinement.initial_fidelity:>7.4f} "
            f"{refinement.refined_fidelity:>8.4f} "
            f"{gain:>+7.4f}"
        )
    if max_rows is not None and len(dataset.targets) > max_rows:
        print(f"... {len(dataset.targets) - max_rows} more")


def print_hamiltonian_solution_dataset_summary(dataset: HamiltonianSolutionDataset) -> None:
    gains = dataset.refined_fidelities - dataset.initial_fidelities
    header = "n   mean before   mean after   median gain   min after   >=0.99 after"
    print(header)
    print("-" * len(header))
    print(
        f"{len(dataset.targets):<3} "
        f"{dataset.initial_fidelities.mean().item():>11.4f}   "
        f"{dataset.refined_fidelities.mean().item():>10.4f}   "
        f"{gains.median().item():>11.4f}   "
        f"{dataset.refined_fidelities.min().item():>9.4f}   "
        f"{(dataset.refined_fidelities >= 0.99).float().mean().item():>10.1%}"
    )


def print_hamiltonian_supervised_summary(result: HamiltonianSupervisedResult) -> None:
    raw = result.raw_fidelities
    refined = None
    if result.refined_results is not None:
        refined = torch.tensor(
            [item.refined_fidelity for item in result.refined_results],
            dtype=torch.float32,
            device=raw.device,
        )

    if refined is None:
        header = "n   mean raw   median   min      max      >=0.95   >=0.99"
        print(header)
        print("-" * len(header))
        print(
            f"{raw.numel():<3} "
            f"{raw.mean().item():>8.4f}   "
            f"{raw.median().item():>6.4f}   "
            f"{raw.min().item():>6.4f}   "
            f"{raw.max().item():>6.4f}   "
            f"{(raw >= 0.95).float().mean().item():>6.1%}   "
            f"{(raw >= 0.99).float().mean().item():>6.1%}"
        )
        return

    gain = refined - raw
    header = "n   mean raw   mean refined   median gain   min refined   >=0.99 refined"
    print(header)
    print("-" * len(header))
    print(
        f"{raw.numel():<3} "
        f"{raw.mean().item():>8.4f}   "
        f"{refined.mean().item():>12.4f}   "
        f"{gain.median().item():>11.4f}   "
        f"{refined.min().item():>11.4f}   "
        f"{(refined >= 0.99).float().mean().item():>14.1%}"
    )


def _supervised_summary_values(result: HamiltonianSupervisedResult) -> tuple[float, float | None, float | None, float, float]:
    raw = result.raw_fidelities
    if result.refined_results is None:
        return (
            float(raw.mean().item()),
            None,
            None,
            float(raw.min().item()),
            float((raw >= 0.99).float().mean().item()),
        )
    refined = torch.tensor(
        [item.refined_fidelity for item in result.refined_results],
        dtype=torch.float32,
        device=raw.device,
    )
    gain = refined - raw
    return (
        float(raw.mean().item()),
        float(refined.mean().item()),
        float(gain.median().item()),
        float(refined.min().item()),
        float((refined >= 0.99).float().mean().item()),
    )


def print_hamiltonian_supervised_split_summary(result: HamiltonianSupervisedSplitResult) -> None:
    header = "split     n   mean raw   mean refined   median gain   min refined   >=0.99 refined"
    print(header)
    print("-" * len(header))
    for split, item in [("train", result.train), ("heldout", result.heldout)]:
        mean_raw, mean_refined, median_gain, min_refined, success_99 = _supervised_summary_values(item)
        if mean_refined is None or median_gain is None:
            mean_refined = mean_raw
            median_gain = 0.0
        print(
            f"{split:<8} {item.raw_fidelities.numel():<3} "
            f"{mean_raw:>8.4f}   "
            f"{mean_refined:>12.4f}   "
            f"{median_gain:>11.4f}   "
            f"{min_refined:>11.4f}   "
            f"{success_99:>14.1%}"
        )


def plot_hamiltonian_supervised_result(result: HamiltonianSupervisedResult) -> None:
    values = [result.raw_fidelities.detach().cpu().tolist()]
    labels = ["raw prediction"]
    if result.refined_results is not None:
        values.append([item.refined_fidelity for item in result.refined_results])
        labels.append("after refinement")

    plt.figure(figsize=(7, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("unitary fidelity")
    plt.title("Hamiltonian supervised stack predictor")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hamiltonian_supervised_split_result(result: HamiltonianSupervisedSplitResult) -> None:
    values = [
        result.train.raw_fidelities.detach().cpu().tolist(),
        result.heldout.raw_fidelities.detach().cpu().tolist(),
    ]
    labels = ["train raw", "heldout raw"]
    if result.train.refined_results is not None and result.heldout.refined_results is not None:
        values.extend(
            [
                [item.refined_fidelity for item in result.train.refined_results],
                [item.refined_fidelity for item in result.heldout.refined_results],
            ]
        )
        labels.extend(["train refined", "heldout refined"])

    plt.figure(figsize=(9, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("unitary fidelity")
    plt.title("Hamiltonian supervised train vs heldout")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hamiltonian_two_entangler_benchmark(benchmark: HamiltonianSynthesisBenchmark) -> None:
    values = [
        [_best(benchmark.clifford_report)],
        [_best(benchmark.analytic_report)],
        [_best(benchmark.generated_report)],
        [_best(benchmark.haar_report)],
    ]
    labels = ["Clifford", "analytic", "generated", "Haar"]
    plt.figure(figsize=(8, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title(f"Hamiltonian target synthesis: {benchmark.target.name}")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hamiltonian_suite(result: HamiltonianSuiteResult) -> None:
    if not result.benchmarks:
        raise ValueError("result must contain at least one benchmark")
    values = [
        [_best(item.clifford_report) for item in result.benchmarks],
        [_best(item.analytic_report) for item in result.benchmarks],
        [_best(item.generated_report) for item in result.benchmarks],
        [_best(item.haar_report) for item in result.benchmarks],
    ]
    labels = ["Clifford", "analytic", "generated", "Haar"]
    plt.figure(figsize=(8, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Hamiltonian target synthesis suite")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hamiltonian_solution_dataset(dataset: HamiltonianSolutionDataset) -> None:
    values = [
        dataset.initial_fidelities.detach().cpu().tolist(),
        dataset.refined_fidelities.detach().cpu().tolist(),
    ]
    plt.figure(figsize=(7, 4))
    plt.boxplot(values, labels=["before refinement", "after refinement"], showmeans=True)
    plt.ylabel("unitary fidelity")
    plt.title("Hamiltonian solution-stack refinement dataset")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()
