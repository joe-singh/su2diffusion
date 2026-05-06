from dataclasses import dataclass, replace
import re

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .circuit import CircuitExperimentConfig, CircuitTrainConfig, circuit_forward_heat_target
from .diffusion import DiffusionSchedule
from .model import (
    SlotwiseTargetConditionedCircuitDenoiser,
    TargetConditionedCircuitDenoiser,
    TargetConditionedCircuitTokenDenoiser,
    TargetLabelConditionedCircuitDenoiser,
)
from .quaternion import q_exp, q_mul, q_normalize, sample_haar, su2_distance
from .synthesis import (
    HiddenShallowCircuitAggregate,
    RefinementResult,
    SynthesisCandidate,
    SynthesisReport,
    compose_local_entangler_chain_units,
    quaternion_to_unitary,
    refine_two_entangler_candidate,
    sample_near_clifford_gates,
    synthesize_unitary_two_entangler_random_report,
    two_qubit_gate,
    unitary_fidelity,
    unitary_fidelity_batch,
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


@dataclass
class HamiltonianConditionedDiffusionResult:
    config: CircuitExperimentConfig
    model: torch.nn.Module
    losses: list[float]
    train_dataset: HamiltonianSolutionDataset
    eval_targets: list[HamiltonianTarget]
    generated_by_target: torch.Tensor
    reports: list[SynthesisReport]


@dataclass
class HamiltonianConditionedOverfitDiagnosticResult:
    config: CircuitExperimentConfig
    model: torch.nn.Module
    losses: list[float]
    train_dataset: HamiltonianSolutionDataset
    train_targets: list[HamiltonianTarget]
    heldout_targets: list[HamiltonianTarget]
    train_generated_by_target: torch.Tensor
    heldout_generated_by_target: torch.Tensor
    train_reports: list[SynthesisReport]
    heldout_reports: list[SynthesisReport]


@dataclass(frozen=True)
class HamiltonianDenoiseDiagnosticRow:
    timestep: int
    sigma: float
    mse: float
    zero_mse: float
    relative_mse: float
    cosine: float
    target_norm: float
    pred_norm: float


@dataclass
class HamiltonianDenoiseDiagnosticResult:
    config: CircuitExperimentConfig
    model: torch.nn.Module
    losses: list[float]
    train_dataset: HamiltonianSolutionDataset
    rows: list[HamiltonianDenoiseDiagnosticRow]


@dataclass(frozen=True)
class HamiltonianDenoiseAblationRow:
    name: str
    num_steps: int
    hidden: int
    final_loss: float
    t1_relative_mse: float
    final_relative_mse: float
    final_cosine: float
    final_pred_target_norm_ratio: float


@dataclass
class HamiltonianDenoiseAblationResult:
    train_dataset: HamiltonianSolutionDataset
    diagnostics: list[HamiltonianDenoiseDiagnosticResult]
    rows: list[HamiltonianDenoiseAblationRow]


@dataclass(frozen=True)
class HamiltonianDenoiseNormalizationRow:
    variant: str
    target_scale: float
    final_loss: float
    final_relative_mse: float
    final_cosine: float
    final_pred_target_norm_ratio: float


@dataclass
class HamiltonianDenoiseNormalizationResult:
    train_dataset: HamiltonianSolutionDataset
    diagnostics: list[HamiltonianDenoiseDiagnosticResult]
    rows: list[HamiltonianDenoiseNormalizationRow]


@dataclass
class HamiltonianSkeletonDenoiseDiagnosticResult:
    config: CircuitExperimentConfig
    model: torch.nn.Module
    losses: list[float]
    train_dataset: HamiltonianSolutionDataset
    label_names: tuple[str, ...]
    rows: list[HamiltonianDenoiseDiagnosticRow]


@dataclass(frozen=True)
class HamiltonianSkeletonDenoiseComparisonRow:
    variant: str
    final_loss: float
    final_relative_mse: float
    final_cosine: float
    final_pred_target_norm_ratio: float


@dataclass
class HamiltonianSkeletonDenoiseComparisonResult:
    unconditioned: HamiltonianDenoiseDiagnosticResult
    skeleton_conditioned: HamiltonianSkeletonDenoiseDiagnosticResult
    rows: list[HamiltonianSkeletonDenoiseComparisonRow]


@dataclass(frozen=True)
class HamiltonianSlotwiseDenoiseComparisonRow:
    variant: str
    final_loss: float
    final_relative_mse: float
    final_cosine: float
    final_pred_target_norm_ratio: float


@dataclass
class HamiltonianSlotwiseDenoiseComparisonResult:
    flat: HamiltonianDenoiseDiagnosticResult
    slotwise: HamiltonianDenoiseDiagnosticResult
    rows: list[HamiltonianSlotwiseDenoiseComparisonRow]


@dataclass(frozen=True)
class HamiltonianTokenDenoiseComparisonRow:
    variant: str
    final_loss: float
    final_relative_mse: float
    final_cosine: float
    final_pred_target_norm_ratio: float


@dataclass
class HamiltonianTokenDenoiseComparisonResult:
    flat: HamiltonianDenoiseDiagnosticResult
    token: HamiltonianDenoiseDiagnosticResult
    rows: list[HamiltonianTokenDenoiseComparisonRow]


@dataclass(frozen=True)
class HamiltonianTokenDataScaleRow:
    n_train_targets: int
    n_solution_stacks: int
    final_loss: float
    train_mean_best: float
    heldout_mean_best: float
    generated_mean_best: float
    heldout_delta_vs_generated: float
    heldout_median_best: float
    heldout_min_best: float
    heldout_max_best: float
    heldout_success_95: float
    heldout_success_98: float
    heldout_success_99: float


@dataclass
class HamiltonianTokenDataScaleResult:
    heldout_targets: list[HamiltonianTarget]
    heldout_baseline: HamiltonianSuiteResult
    diagnostics: dict[int, HamiltonianConditionedOverfitDiagnosticResult]
    rows: list[HamiltonianTokenDataScaleRow]


@dataclass(frozen=True)
class HamiltonianTokenTrainingBudgetRow:
    num_steps: int
    hidden: int
    batch_size: int
    n_train_targets: int
    n_solution_stacks: int
    final_loss: float
    train_mean_best: float
    heldout_mean_best: float
    generated_mean_best: float
    heldout_delta_vs_generated: float
    heldout_median_best: float
    heldout_min_best: float
    heldout_max_best: float
    heldout_success_95: float
    heldout_success_98: float
    heldout_success_99: float


@dataclass
class HamiltonianTokenTrainingBudgetResult:
    train_dataset: HamiltonianSolutionDataset
    heldout_targets: list[HamiltonianTarget]
    heldout_baseline: HamiltonianSuiteResult
    diagnostics: dict[int, HamiltonianConditionedOverfitDiagnosticResult]
    rows: list[HamiltonianTokenTrainingBudgetRow]


@dataclass(frozen=True)
class HamiltonianTokenRepeatabilityRow:
    run: int
    train_seed: int
    heldout_seed: int
    dataset_seed: int
    baseline_seed: int
    num_steps: int
    n_train_targets: int
    n_heldout_targets: int
    n_solution_stacks: int
    final_loss: float
    train_mean_best: float
    heldout_mean_best: float
    generated_mean_best: float
    heldout_delta_vs_generated: float
    heldout_success_95: float
    heldout_success_98: float
    heldout_success_99: float


@dataclass
class HamiltonianTokenRepeatabilityResult:
    budget_results: list[HamiltonianTokenTrainingBudgetResult]
    rows: list[HamiltonianTokenRepeatabilityRow]


@dataclass(frozen=True)
class HamiltonianRepeatabilityRefinementRow:
    run: int
    target: str
    source: str
    initial_fidelity: float
    refined_fidelity: float
    steps_to_threshold: int
    slot_movements: tuple[float, ...]
    movement_mean: float
    movement_max: float


@dataclass
class HamiltonianRepeatabilityRefinementResult:
    repeatability: HamiltonianTokenRepeatabilityResult
    rows: list[HamiltonianRepeatabilityRefinementRow]
    threshold: float


@dataclass(frozen=True)
class HamiltonianLevel1HeadlineRow:
    source: str
    n_targets: int
    proposal_mean: float
    proposal_run_std: float
    refined_mean: float
    refinement_success: float
    median_steps: float
    mean_movement: float
    max_movement: float


@dataclass(frozen=True)
class HamiltonianLevel1HeadlineResult:
    rows: list[HamiltonianLevel1HeadlineRow]
    n_runs: int
    threshold: float
    proposal_advantage_mean: float
    proposal_advantage_std: float


@dataclass(frozen=True)
class HamiltonianTemplateComparisonRow:
    template: str
    n_entanglers: int
    n_slots: int
    n_stacks: int
    proposal_mean: float
    refined_mean: float
    refinement_success: float
    median_steps: float


@dataclass(frozen=True)
class HamiltonianTemplateComparisonResult:
    two_entangler: HamiltonianSolutionDataset
    three_entangler: HamiltonianSolutionDataset
    rows: list[HamiltonianTemplateComparisonRow]
    threshold: float


@dataclass(frozen=True)
class HamiltonianTokenTemplateComparisonRow:
    template: str
    n_entanglers: int
    n_slots: int
    n_train_targets: int
    n_heldout_targets: int
    n_solution_stacks: int
    num_steps: int
    final_loss: float
    train_mean_best: float
    heldout_mean_best: float
    generated_mean_best: float
    heldout_delta_vs_generated: float
    heldout_success_95: float
    heldout_success_98: float
    heldout_success_99: float


@dataclass
class HamiltonianTokenTemplateComparisonResult:
    two_entangler: "HamiltonianTokenTrainingBudgetResult"
    three_entangler: "HamiltonianTokenTrainingBudgetResult"
    rows: list[HamiltonianTokenTemplateComparisonRow]


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


@dataclass(frozen=True)
class HamiltonianSeedAblationRow:
    target: str
    seed_type: str
    initial_fidelity: float
    refined_fidelity: float
    steps_to_threshold: int


@dataclass(frozen=True)
class HamiltonianSeedAblationResult:
    rows: list[HamiltonianSeedAblationRow]
    threshold: float


@dataclass(frozen=True)
class HamiltonianPriorTrainConfig:
    hidden: int = 128
    num_steps: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


@dataclass
class HamiltonianPriorResult:
    model: torch.nn.Module
    losses: list[float]
    label_names: tuple[str, ...]
    train_accuracy: float


@dataclass(frozen=True)
class HamiltonianPriorSearchBenchmark:
    target: HamiltonianTarget
    uniform_report: SynthesisReport
    prior_report: SynthesisReport


@dataclass(frozen=True)
class HamiltonianPriorSearchResult:
    benchmarks: list[HamiltonianPriorSearchBenchmark]


@dataclass(frozen=True)
class HamiltonianPriorMixtureResult:
    alpha_results: dict[float, HamiltonianPriorSearchResult]


@dataclass(frozen=True)
class HamiltonianMixtureRefinementRow:
    target: str
    alpha: float
    initial_fidelity: float
    refined_fidelity: float
    steps_to_threshold: int


@dataclass(frozen=True)
class HamiltonianMixtureRefinementResult:
    rows: list[HamiltonianMixtureRefinementRow]
    threshold: float


@dataclass(frozen=True)
class HamiltonianBudgetRefinementRow:
    budget: int
    target: str
    alpha: float
    initial_fidelity: float
    refined_fidelity: float
    reached_threshold: bool


@dataclass(frozen=True)
class HamiltonianBudgetRefinementResult:
    rows: list[HamiltonianBudgetRefinementRow]
    threshold: float


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


class HamiltonianSlotPriorPredictor(nn.Module):
    def __init__(self, input_dim: int = 33, hidden: int = 128, n_slots: int = 6, n_labels: int = 24):
        super().__init__()
        self.input_dim = input_dim
        self.n_slots = n_slots
        self.n_labels = n_labels
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_slots * n_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"Expected features with shape (batch, {self.input_dim})")
        return self.net(features).reshape(features.shape[0], self.n_slots, self.n_labels)


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
    n_entanglers: int = 2,
    n_random_candidates: int = 200_000,
    n_analytic_gates: int = 1024,
    n_haar_gates: int = 1024,
    top_k: int = 5,
    seed: int = 0,
    keep_fidelities: bool = True,
) -> HamiltonianSynthesisBenchmark:
    if n_entanglers <= 0:
        raise ValueError("n_entanglers must be positive")
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
        n_entanglers=n_entanglers,
        n_candidates=n_random_candidates,
        top_k=top_k,
        local_labels=clifford_labels,
        seed=seed + 10_000,
        name=f"{target.name} Clifford {n_entanglers}-entangler random",
        keep_fidelities=keep_fidelities,
    )
    analytic_report = synthesize_unitary_two_entangler_random_report(
        analytic_gates,
        target_unitary=target_unitary,
        target_name=target.name,
        entangler=entangler,
        n_entanglers=n_entanglers,
        n_candidates=n_random_candidates,
        top_k=top_k,
        local_labels=analytic_labels,
        seed=seed + 15_000,
        name=f"{target.name} analytic near-Clifford {n_entanglers}-entangler random",
        keep_fidelities=keep_fidelities,
    )
    generated_report = synthesize_unitary_two_entangler_random_report(
        generated_gates,
        target_unitary=target_unitary,
        target_name=target.name,
        entangler=entangler,
        n_entanglers=n_entanglers,
        n_candidates=n_random_candidates,
        top_k=top_k,
        local_labels=generated_labels,
        seed=seed + 20_000,
        name=f"{target.name} generated {n_entanglers}-entangler random",
        keep_fidelities=keep_fidelities,
    )
    haar_report = synthesize_unitary_two_entangler_random_report(
        haar_gates,
        target_unitary=target_unitary,
        target_name=target.name,
        entangler=entangler,
        n_entanglers=n_entanglers,
        n_candidates=n_random_candidates,
        top_k=top_k,
        local_labels=haar_labels,
        seed=seed + 40_000,
        name=f"{target.name} Haar {n_entanglers}-entangler random",
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
    n_entanglers: int = 2,
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
            n_entanglers=n_entanglers,
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
    n_entanglers: int = 2,
    n_random_candidates: int = 100_000,
    n_analytic_gates: int = 1024,
    n_haar_gates: int = 1024,
    top_k: int = 5,
    seed: int = 0,
    refinement_steps: int = 200,
    refinement_lr: float = 0.05,
    fidelity_threshold: float = 0.0,
    solutions_per_target: int = 1,
) -> HamiltonianSolutionDataset:
    if not targets:
        raise ValueError("targets must contain at least one Hamiltonian target")
    if refinement_steps <= 0:
        raise ValueError("refinement_steps must be positive")
    if refinement_lr <= 0:
        raise ValueError("refinement_lr must be positive")
    if not (0.0 <= fidelity_threshold <= 1.0):
        raise ValueError("fidelity_threshold must be between 0 and 1")
    if solutions_per_target <= 0:
        raise ValueError("solutions_per_target must be positive")

    suite = run_hamiltonian_suite_benchmark(
        targets,
        clifford_gates=clifford_gates,
        clifford_labels=clifford_labels,
        generated_gates=generated_gates,
        generated_labels=generated_labels,
        perturb_scale=perturb_scale,
        entangler=entangler,
        n_entanglers=n_entanglers,
        n_random_candidates=n_random_candidates,
        n_analytic_gates=n_analytic_gates,
        n_haar_gates=n_haar_gates,
        top_k=max(top_k, solutions_per_target),
        seed=seed,
        keep_fidelities=False,
    )

    kept_targets = []
    kept_benchmarks = []
    refinements = []
    for benchmark in suite.benchmarks:
        for candidate in benchmark.generated_report.candidates[:solutions_per_target]:
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
    return compose_local_entangler_chain_units(units, entangler_unitary)


def _stack_fidelity(q_stack: torch.Tensor, target: HamiltonianTarget, entangler: str = "cz") -> float:
    return unitary_fidelity(_stack_unitary(q_stack, entangler=entangler), target.unitary)


def _candidate_from_stack(
    target: HamiltonianTarget,
    seed_type: str,
    initial_fidelity: float,
    n_slots: int = 6,
    entangler: str = "cz",
) -> SynthesisCandidate:
    return SynthesisCandidate(
        target=target.name,
        template=f"hamiltonian-{seed_type}-stack",
        entangler=entangler,
        fidelity=initial_fidelity,
        slot_indices=tuple(range(n_slots)),
        slot_labels=(seed_type,) * n_slots,
    )


def _steps_to_threshold(initial_fidelity: float, trace: tuple[float, ...], threshold: float) -> int:
    if initial_fidelity >= threshold:
        return 0
    for i, value in enumerate(trace, start=1):
        if value >= threshold:
            return i
    return -1


def _refinement_movement(
    start_gates: torch.Tensor,
    refined_gates: torch.Tensor,
) -> tuple[tuple[float, ...], float, float]:
    movement = su2_distance(q_normalize(start_gates), q_normalize(refined_gates)).detach().cpu()
    slot_movements = tuple(float(value) for value in movement.tolist())
    return slot_movements, float(movement.mean().item()), float(movement.max().item())


def _validate_solution_stacks(stacks: torch.Tensor, name: str = "dataset.stacks") -> int:
    if stacks.ndim != 3 or stacks.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (n, n_slots, 4)")
    if stacks.shape[1] < 4 or stacks.shape[1] % 2 != 0:
        raise ValueError(f"{name} must contain an even number of slots, at least 4")
    return int(stacks.shape[1])


def _template_comparison_row(
    template: str,
    dataset: HamiltonianSolutionDataset,
    threshold: float,
) -> HamiltonianTemplateComparisonRow:
    n_slots = _validate_solution_stacks(dataset.stacks)
    steps = []
    for refinement in dataset.refinements:
        step = _steps_to_threshold(
            refinement.initial_fidelity,
            refinement.fidelity_trace,
            threshold,
        )
        if step < 0:
            step = len(refinement.fidelity_trace) + 1
        steps.append(step)
    step_tensor = torch.tensor(steps, dtype=torch.float32, device=dataset.refined_fidelities.device)
    return HamiltonianTemplateComparisonRow(
        template=template,
        n_entanglers=n_slots // 2 - 1,
        n_slots=n_slots,
        n_stacks=len(dataset.targets),
        proposal_mean=float(dataset.initial_fidelities.mean().item()),
        refined_mean=float(dataset.refined_fidelities.mean().item()),
        refinement_success=float((dataset.refined_fidelities >= threshold).float().mean().item()),
        median_steps=float(step_tensor.median().item()),
    )


def run_hamiltonian_template_comparison(
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
    refinement_steps: int = 50,
    refinement_lr: float = 0.05,
    threshold: float = 0.99,
    fidelity_threshold: float = 0.0,
    solutions_per_target: int = 1,
) -> HamiltonianTemplateComparisonResult:
    if not targets:
        raise ValueError("targets must contain at least one Hamiltonian target")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    two_entangler = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=clifford_gates,
        clifford_labels=clifford_labels,
        generated_gates=generated_gates,
        generated_labels=generated_labels,
        perturb_scale=perturb_scale,
        entangler=entangler,
        n_entanglers=2,
        n_random_candidates=n_random_candidates,
        n_analytic_gates=n_analytic_gates,
        n_haar_gates=n_haar_gates,
        top_k=top_k,
        seed=seed,
        refinement_steps=refinement_steps,
        refinement_lr=refinement_lr,
        fidelity_threshold=fidelity_threshold,
        solutions_per_target=solutions_per_target,
    )
    three_entangler = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=clifford_gates,
        clifford_labels=clifford_labels,
        generated_gates=generated_gates,
        generated_labels=generated_labels,
        perturb_scale=perturb_scale,
        entangler=entangler,
        n_entanglers=3,
        n_random_candidates=n_random_candidates,
        n_analytic_gates=n_analytic_gates,
        n_haar_gates=n_haar_gates,
        top_k=top_k,
        seed=seed + 100_000,
        refinement_steps=refinement_steps,
        refinement_lr=refinement_lr,
        fidelity_threshold=fidelity_threshold,
        solutions_per_target=solutions_per_target,
    )
    return HamiltonianTemplateComparisonResult(
        two_entangler=two_entangler,
        three_entangler=three_entangler,
        rows=[
            _template_comparison_row("2 CZ / 6 local gates", two_entangler, threshold),
            _template_comparison_row("3 CZ / 8 local gates", three_entangler, threshold),
        ],
        threshold=threshold,
    )


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
            candidate = _candidate_from_stack(target, "predicted", fidelity, n_slots=stack.shape[0], entangler=entangler)
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


def _eps_output_scale(model: torch.nn.Module) -> float:
    return float(getattr(model, "eps_output_scale", 1.0))


def _predict_hamiltonian_eps(
    model: TargetConditionedCircuitDenoiser,
    q_stack: torch.Tensor,
    t_idx: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    return model(q_stack, t_idx, features) * _eps_output_scale(model)


def _predict_hamiltonian_skeleton_eps(
    model: TargetLabelConditionedCircuitDenoiser,
    q_stack: torch.Tensor,
    t_idx: torch.Tensor,
    features: torch.Tensor,
    slot_labels: torch.Tensor,
) -> torch.Tensor:
    return model(q_stack, t_idx, features, slot_labels) * _eps_output_scale(model)


def estimate_hamiltonian_denoise_target_scale(
    dataset: HamiltonianSolutionDataset,
    schedule: DiffusionSchedule,
    batch_size: int = 512,
    n_batches: int = 8,
    n_terms: int = 128,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> float:
    if not dataset.targets:
        raise ValueError("dataset must contain at least one Hamiltonian target")
    _validate_solution_stacks(dataset.stacks)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")

    device = torch.device(device) if device is not None else dataset.stacks.device
    stacks = q_normalize(dataset.stacks.to(device=device))
    sq_means = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _ in range(n_batches):
            rows = torch.randint(
                low=0,
                high=stacks.shape[0],
                size=(batch_size,),
                device=device,
            )
            t_idx = torch.randint(1, schedule.T + 1, (batch_size,), device=device)
            _, eps_target = circuit_forward_heat_target(
                stacks[rows],
                t_idx,
                schedule=schedule,
                n_terms=n_terms,
            )
            sq_means.append(eps_target.square().mean())
    scale = torch.stack(sq_means).mean().sqrt().item()
    return float(max(scale, 1e-6))


def train_hamiltonian_conditioned_circuit_diffusion(
    dataset: HamiltonianSolutionDataset,
    train_config: CircuitTrainConfig | None = None,
    schedule: DiffusionSchedule | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    target_scale: float = 1.0,
) -> tuple[TargetConditionedCircuitDenoiser, list[float]]:
    if not dataset.targets:
        raise ValueError("dataset must contain at least one Hamiltonian target")
    n_slots = _validate_solution_stacks(dataset.stacks)
    if dataset.stacks.shape[0] != len(dataset.targets):
        raise ValueError("dataset.stacks must contain one stack per Hamiltonian target")

    train_config = train_config or CircuitTrainConfig()
    schedule = schedule or DiffusionSchedule()
    if target_scale <= 0:
        raise ValueError("target_scale must be positive")
    device = torch.device(device) if device is not None else dataset.stacks.device
    stacks = q_normalize(dataset.stacks.to(device=device))
    features = hamiltonian_target_features(dataset.targets).to(device=device)

    torch.manual_seed(train_config.seed)
    model = TargetConditionedCircuitDenoiser(
        T=schedule.T,
        n_slots=n_slots,
        target_dim=features.shape[1],
        hidden=train_config.hidden,
    ).to(device)
    model.eps_output_scale = float(target_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    losses: list[float] = []
    iterator = range(1, train_config.num_steps + 1)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Training Hamiltonian-conditioned circuit diffusion", dynamic_ncols=True)

    for _ in iterator:
        rows = torch.randint(
            low=0,
            high=stacks.shape[0],
            size=(train_config.batch_size,),
            device=device,
        )
        q0_stack = stacks[rows]
        batch_features = features[rows]
        t_idx = torch.randint(1, schedule.T + 1, (train_config.batch_size,), device=device)

        with torch.no_grad():
            qt_stack, eps_target = circuit_forward_heat_target(
                q0_stack,
                t_idx,
                schedule=schedule,
                n_terms=train_config.n_terms,
            )

        eps_pred = model(qt_stack, t_idx, batch_features)
        loss = F.mse_loss(eps_pred, eps_target / target_scale)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_value:.5f}"})

    return model, losses


def train_hamiltonian_skeleton_conditioned_circuit_diffusion(
    dataset: HamiltonianSolutionDataset,
    label_names: tuple[str, ...] | list[str],
    train_config: CircuitTrainConfig | None = None,
    schedule: DiffusionSchedule | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    target_scale: float = 1.0,
) -> tuple[TargetLabelConditionedCircuitDenoiser, list[float]]:
    if not dataset.targets:
        raise ValueError("dataset must contain at least one Hamiltonian target")
    n_slots = _validate_solution_stacks(dataset.stacks)
    if dataset.stacks.shape[0] != len(dataset.targets):
        raise ValueError("dataset.stacks must contain one stack per Hamiltonian target")

    train_config = train_config or CircuitTrainConfig()
    schedule = schedule or DiffusionSchedule()
    if target_scale <= 0:
        raise ValueError("target_scale must be positive")
    device = torch.device(device) if device is not None else dataset.stacks.device
    stacks = q_normalize(dataset.stacks.to(device=device))
    features = hamiltonian_target_features(dataset.targets).to(device=device)
    slot_labels = _slot_label_targets(dataset, label_names, device=device)

    torch.manual_seed(train_config.seed)
    model = TargetLabelConditionedCircuitDenoiser(
        T=schedule.T,
        n_slots=n_slots,
        target_dim=features.shape[1],
        num_labels=len(label_names),
        hidden=train_config.hidden,
    ).to(device)
    model.eps_output_scale = float(target_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    losses: list[float] = []
    iterator = range(1, train_config.num_steps + 1)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Training Hamiltonian skeleton-conditioned diffusion", dynamic_ncols=True)

    for _ in iterator:
        rows = torch.randint(
            low=0,
            high=stacks.shape[0],
            size=(train_config.batch_size,),
            device=device,
        )
        q0_stack = stacks[rows]
        batch_features = features[rows]
        batch_labels = slot_labels[rows]
        t_idx = torch.randint(1, schedule.T + 1, (train_config.batch_size,), device=device)

        with torch.no_grad():
            qt_stack, eps_target = circuit_forward_heat_target(
                q0_stack,
                t_idx,
                schedule=schedule,
                n_terms=train_config.n_terms,
            )

        eps_pred = model(qt_stack, t_idx, batch_features, batch_labels)
        loss = F.mse_loss(eps_pred, eps_target / target_scale)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_value:.5f}"})

    return model, losses


def train_hamiltonian_slotwise_circuit_diffusion(
    dataset: HamiltonianSolutionDataset,
    train_config: CircuitTrainConfig | None = None,
    schedule: DiffusionSchedule | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    target_scale: float = 1.0,
) -> tuple[SlotwiseTargetConditionedCircuitDenoiser, list[float]]:
    if not dataset.targets:
        raise ValueError("dataset must contain at least one Hamiltonian target")
    n_slots = _validate_solution_stacks(dataset.stacks)
    if dataset.stacks.shape[0] != len(dataset.targets):
        raise ValueError("dataset.stacks must contain one stack per Hamiltonian target")

    train_config = train_config or CircuitTrainConfig()
    schedule = schedule or DiffusionSchedule()
    if target_scale <= 0:
        raise ValueError("target_scale must be positive")
    device = torch.device(device) if device is not None else dataset.stacks.device
    stacks = q_normalize(dataset.stacks.to(device=device))
    features = hamiltonian_target_features(dataset.targets).to(device=device)

    torch.manual_seed(train_config.seed)
    model = SlotwiseTargetConditionedCircuitDenoiser(
        T=schedule.T,
        n_slots=n_slots,
        target_dim=features.shape[1],
        hidden=train_config.hidden,
    ).to(device)
    model.eps_output_scale = float(target_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    losses: list[float] = []
    iterator = range(1, train_config.num_steps + 1)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Training Hamiltonian slot-wise circuit diffusion", dynamic_ncols=True)

    for _ in iterator:
        rows = torch.randint(
            low=0,
            high=stacks.shape[0],
            size=(train_config.batch_size,),
            device=device,
        )
        q0_stack = stacks[rows]
        batch_features = features[rows]
        t_idx = torch.randint(1, schedule.T + 1, (train_config.batch_size,), device=device)

        with torch.no_grad():
            qt_stack, eps_target = circuit_forward_heat_target(
                q0_stack,
                t_idx,
                schedule=schedule,
                n_terms=train_config.n_terms,
            )

        eps_pred = model(qt_stack, t_idx, batch_features)
        loss = F.mse_loss(eps_pred, eps_target / target_scale)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_value:.5f}"})

    return model, losses


def train_hamiltonian_token_circuit_diffusion(
    dataset: HamiltonianSolutionDataset,
    train_config: CircuitTrainConfig | None = None,
    schedule: DiffusionSchedule | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    target_scale: float = 1.0,
) -> tuple[TargetConditionedCircuitTokenDenoiser, list[float]]:
    if not dataset.targets:
        raise ValueError("dataset must contain at least one Hamiltonian target")
    n_slots = _validate_solution_stacks(dataset.stacks)
    if dataset.stacks.shape[0] != len(dataset.targets):
        raise ValueError("dataset.stacks must contain one stack per Hamiltonian target")

    train_config = train_config or CircuitTrainConfig()
    schedule = schedule or DiffusionSchedule()
    if target_scale <= 0:
        raise ValueError("target_scale must be positive")
    device = torch.device(device) if device is not None else dataset.stacks.device
    stacks = q_normalize(dataset.stacks.to(device=device))
    features = hamiltonian_target_features(dataset.targets).to(device=device)

    torch.manual_seed(train_config.seed)
    model = TargetConditionedCircuitTokenDenoiser(
        T=schedule.T,
        n_slots=n_slots,
        target_dim=features.shape[1],
        hidden=train_config.hidden,
    ).to(device)
    model.eps_output_scale = float(target_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    losses: list[float] = []
    iterator = range(1, train_config.num_steps + 1)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Training Hamiltonian circuit-token diffusion", dynamic_ncols=True)

    for _ in iterator:
        rows = torch.randint(
            low=0,
            high=stacks.shape[0],
            size=(train_config.batch_size,),
            device=device,
        )
        q0_stack = stacks[rows]
        batch_features = features[rows]
        t_idx = torch.randint(1, schedule.T + 1, (train_config.batch_size,), device=device)

        with torch.no_grad():
            qt_stack, eps_target = circuit_forward_heat_target(
                q0_stack,
                t_idx,
                schedule=schedule,
                n_terms=train_config.n_terms,
            )

        eps_pred = model(qt_stack, t_idx, batch_features)
        loss = F.mse_loss(eps_pred, eps_target / target_scale)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_value:.5f}"})

    return model, losses


@torch.no_grad()
def sample_hamiltonian_conditioned_circuit_reverse(
    model: TargetConditionedCircuitDenoiser,
    schedule: DiffusionSchedule,
    targets: list[HamiltonianTarget],
    n_samples_per_target: int = 1000,
    eta: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not targets:
        raise ValueError("targets must contain at least one Hamiltonian target")
    if n_samples_per_target <= 0:
        raise ValueError("n_samples_per_target must be positive")

    device = torch.device(device) if device is not None else next(model.parameters()).device
    n_targets = len(targets)
    n_slots = getattr(model, "n_slots", 6)
    n_total = n_targets * n_samples_per_target
    features = hamiltonian_target_features(targets).to(device=device)
    if features.shape[1] != model.target_dim:
        raise ValueError(f"Model expects {model.target_dim} Hamiltonian features, got {features.shape[1]}")
    features = features[:, None, :].expand(n_targets, n_samples_per_target, features.shape[-1])
    features = features.reshape(n_total, features.shape[-1])

    betas, _, sigmas = schedule.tensors(device)
    q_stack = sample_haar(n_total * n_slots, device=device).reshape(n_total, n_slots, 4)

    for s in reversed(range(schedule.T)):
        t_idx = torch.full((n_total,), s + 1, device=device, dtype=torch.long)
        eps_pred = _predict_hamiltonian_eps(model, q_stack, t_idx, features)

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


def run_hamiltonian_conditioned_diffusion_benchmark(
    train_dataset: HamiltonianSolutionDataset,
    eval_targets: list[HamiltonianTarget],
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    entangler: str = "cz",
    top_k: int = 5,
) -> HamiltonianConditionedDiffusionResult:
    from .circuit import get_circuit_experiment_config, synthesize_unitary_from_circuit_stack_report

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if not eval_targets:
        raise ValueError("eval_targets must contain at least one Hamiltonian target")
    device = torch.device(device) if device is not None else train_dataset.stacks.device

    model, losses = train_hamiltonian_conditioned_circuit_diffusion(
        train_dataset,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
    )
    generated_by_target = sample_hamiltonian_conditioned_circuit_reverse(
        model,
        config.schedule,
        eval_targets,
        n_samples_per_target=config.sample_count,
        eta=config.eta,
        device=device,
    )
    reports = [
        synthesize_unitary_from_circuit_stack_report(
            stacks,
            target_unitary=target.unitary,
            target_name=target.name,
            entangler=entangler,
            top_k=top_k,
            name=f"{target.name} Hamiltonian-conditioned diffusion",
            keep_fidelities=False,
        )
        for target, stacks in zip(eval_targets, generated_by_target)
    ]
    return HamiltonianConditionedDiffusionResult(
        config=config,
        model=model,
        losses=losses,
        train_dataset=train_dataset,
        eval_targets=eval_targets,
        generated_by_target=generated_by_target,
        reports=reports,
    )


def run_hamiltonian_token_conditioned_diffusion_benchmark(
    train_dataset: HamiltonianSolutionDataset,
    eval_targets: list[HamiltonianTarget],
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    entangler: str = "cz",
    top_k: int = 5,
) -> HamiltonianConditionedDiffusionResult:
    from .circuit import get_circuit_experiment_config, synthesize_unitary_from_circuit_stack_report

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if not eval_targets:
        raise ValueError("eval_targets must contain at least one Hamiltonian target")
    device = torch.device(device) if device is not None else train_dataset.stacks.device

    model, losses = train_hamiltonian_token_circuit_diffusion(
        train_dataset,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
    )
    generated_by_target = sample_hamiltonian_conditioned_circuit_reverse(
        model,
        config.schedule,
        eval_targets,
        n_samples_per_target=config.sample_count,
        eta=config.eta,
        device=device,
    )
    reports = [
        synthesize_unitary_from_circuit_stack_report(
            stacks,
            target_unitary=target.unitary,
            target_name=target.name,
            entangler=entangler,
            top_k=top_k,
            name=f"{target.name} Hamiltonian circuit-token diffusion",
            keep_fidelities=False,
        )
        for target, stacks in zip(eval_targets, generated_by_target)
    ]
    return HamiltonianConditionedDiffusionResult(
        config=replace(config, name=f"{config.name}-token"),
        model=model,
        losses=losses,
        train_dataset=train_dataset,
        eval_targets=eval_targets,
        generated_by_target=generated_by_target,
        reports=reports,
    )


def _unique_hamiltonian_targets(targets: list[HamiltonianTarget]) -> list[HamiltonianTarget]:
    unique: dict[str, HamiltonianTarget] = {}
    for target in targets:
        unique.setdefault(target.name, target)
    return list(unique.values())


def _hamiltonian_conditioned_reports_from_batches(
    generated_by_target: torch.Tensor,
    targets: list[HamiltonianTarget],
    prefix: str,
    entangler: str,
    top_k: int,
) -> list[SynthesisReport]:
    from .circuit import synthesize_unitary_from_circuit_stack_report

    if generated_by_target.shape[0] != len(targets):
        raise ValueError("generated_by_target must contain one batch per target")
    return [
        synthesize_unitary_from_circuit_stack_report(
            stacks,
            target_unitary=target.unitary,
            target_name=target.name,
            entangler=entangler,
            top_k=top_k,
            name=f"{prefix} {target.name} Hamiltonian-conditioned diffusion",
            keep_fidelities=False,
        )
        for target, stacks in zip(targets, generated_by_target)
    ]


def run_hamiltonian_conditioned_overfit_diagnostic(
    train_dataset: HamiltonianSolutionDataset,
    heldout_targets: list[HamiltonianTarget],
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    entangler: str = "cz",
    top_k: int = 5,
) -> HamiltonianConditionedOverfitDiagnosticResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if not heldout_targets:
        raise ValueError("heldout_targets must contain at least one target")
    device = torch.device(device) if device is not None else train_dataset.stacks.device
    train_targets = _unique_hamiltonian_targets(train_dataset.targets)

    model, losses = train_hamiltonian_conditioned_circuit_diffusion(
        train_dataset,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
    )
    train_generated = sample_hamiltonian_conditioned_circuit_reverse(
        model,
        config.schedule,
        train_targets,
        n_samples_per_target=config.sample_count,
        eta=config.eta,
        device=device,
    )
    heldout_generated = sample_hamiltonian_conditioned_circuit_reverse(
        model,
        config.schedule,
        heldout_targets,
        n_samples_per_target=config.sample_count,
        eta=config.eta,
        device=device,
    )
    train_reports = _hamiltonian_conditioned_reports_from_batches(
        train_generated,
        train_targets,
        prefix="train",
        entangler=entangler,
        top_k=top_k,
    )
    heldout_reports = _hamiltonian_conditioned_reports_from_batches(
        heldout_generated,
        heldout_targets,
        prefix="heldout",
        entangler=entangler,
        top_k=top_k,
    )
    return HamiltonianConditionedOverfitDiagnosticResult(
        config=config,
        model=model,
        losses=losses,
        train_dataset=train_dataset,
        train_targets=train_targets,
        heldout_targets=heldout_targets,
        train_generated_by_target=train_generated,
        heldout_generated_by_target=heldout_generated,
        train_reports=train_reports,
        heldout_reports=heldout_reports,
    )


def run_hamiltonian_token_conditioned_overfit_diagnostic(
    train_dataset: HamiltonianSolutionDataset,
    heldout_targets: list[HamiltonianTarget],
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    entangler: str = "cz",
    top_k: int = 5,
) -> HamiltonianConditionedOverfitDiagnosticResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if not heldout_targets:
        raise ValueError("heldout_targets must contain at least one target")
    device = torch.device(device) if device is not None else train_dataset.stacks.device
    train_targets = _unique_hamiltonian_targets(train_dataset.targets)

    model, losses = train_hamiltonian_token_circuit_diffusion(
        train_dataset,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
    )
    train_generated = sample_hamiltonian_conditioned_circuit_reverse(
        model,
        config.schedule,
        train_targets,
        n_samples_per_target=config.sample_count,
        eta=config.eta,
        device=device,
    )
    heldout_generated = sample_hamiltonian_conditioned_circuit_reverse(
        model,
        config.schedule,
        heldout_targets,
        n_samples_per_target=config.sample_count,
        eta=config.eta,
        device=device,
    )
    train_reports = _hamiltonian_conditioned_reports_from_batches(
        train_generated,
        train_targets,
        prefix="train token",
        entangler=entangler,
        top_k=top_k,
    )
    heldout_reports = _hamiltonian_conditioned_reports_from_batches(
        heldout_generated,
        heldout_targets,
        prefix="heldout token",
        entangler=entangler,
        top_k=top_k,
    )
    return HamiltonianConditionedOverfitDiagnosticResult(
        config=replace(config, name=f"{config.name}-token-overfit"),
        model=model,
        losses=losses,
        train_dataset=train_dataset,
        train_targets=train_targets,
        heldout_targets=heldout_targets,
        train_generated_by_target=train_generated,
        heldout_generated_by_target=heldout_generated,
        train_reports=train_reports,
        heldout_reports=heldout_reports,
    )


def run_hamiltonian_token_data_scale_benchmark(
    train_target_counts: tuple[int, ...] | list[int],
    heldout_targets: list[HamiltonianTarget],
    clifford_gates: torch.Tensor,
    clifford_labels: list[str],
    generated_gates: torch.Tensor,
    generated_labels: list[str],
    config: CircuitExperimentConfig | str,
    terms: tuple[str, ...] = ("XI", "IZ", "XX", "ZZ"),
    coefficient_scale: float = 0.35,
    time: float = 0.8,
    train_seed: int = 2607,
    dataset_seed: int = 2707,
    heldout_baseline_seed: int = 2807,
    perturb_scale: float = 0.12,
    entangler: str = "cz",
    n_entanglers: int = 2,
    n_random_candidates: int = 25_000,
    n_analytic_gates: int = 1024,
    n_haar_gates: int = 1024,
    top_k: int = 5,
    refinement_steps: int = 160,
    refinement_lr: float = 0.05,
    fidelity_threshold: float = 0.0,
    solutions_per_target: int = 2,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> HamiltonianTokenDataScaleResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if not train_target_counts:
        raise ValueError("train_target_counts must contain at least one count")
    counts = tuple(sorted({int(count) for count in train_target_counts}))
    if counts[0] <= 0:
        raise ValueError("train target counts must be positive")
    if not heldout_targets:
        raise ValueError("heldout_targets must contain at least one target")

    device = torch.device(device) if device is not None else generated_gates.device
    train_pool = make_random_pauli_hamiltonian_targets(
        n_targets=max(counts),
        terms=terms,
        coefficient_scale=coefficient_scale,
        time=time,
        seed=train_seed,
        device=device,
    )
    heldout_baseline = run_hamiltonian_suite_benchmark(
        heldout_targets,
        clifford_gates=clifford_gates,
        clifford_labels=clifford_labels,
        generated_gates=generated_gates,
        generated_labels=generated_labels,
        perturb_scale=perturb_scale,
        entangler=entangler,
        n_entanglers=n_entanglers,
        n_random_candidates=n_random_candidates,
        n_analytic_gates=n_analytic_gates,
        n_haar_gates=n_haar_gates,
        top_k=top_k,
        seed=heldout_baseline_seed,
        keep_fidelities=False,
    )
    generated_baseline = _aggregate_reports(
        "generated random",
        [item.generated_report for item in heldout_baseline.benchmarks],
    )

    diagnostics: dict[int, HamiltonianConditionedOverfitDiagnosticResult] = {}
    rows: list[HamiltonianTokenDataScaleRow] = []
    for i, count in enumerate(counts):
        train_targets = train_pool[:count]
        dataset = generate_hamiltonian_solution_dataset(
            train_targets,
            clifford_gates=clifford_gates,
            clifford_labels=clifford_labels,
            generated_gates=generated_gates,
            generated_labels=generated_labels,
            perturb_scale=perturb_scale,
            entangler=entangler,
            n_entanglers=n_entanglers,
            n_random_candidates=n_random_candidates,
            n_analytic_gates=n_analytic_gates,
            n_haar_gates=n_haar_gates,
            top_k=max(top_k, solutions_per_target),
            seed=dataset_seed + i,
            refinement_steps=refinement_steps,
            refinement_lr=refinement_lr,
            fidelity_threshold=fidelity_threshold,
            solutions_per_target=solutions_per_target,
        )
        diagnostic = run_hamiltonian_token_conditioned_overfit_diagnostic(
            dataset,
            heldout_targets=heldout_targets,
            config=replace(config, name=f"{config.name}-n{count}"),
            device=device,
            show_progress=show_progress,
            entangler=entangler,
            top_k=top_k,
        )
        diagnostics[count] = diagnostic

        train_summary = _aggregate_reports("train targets", diagnostic.train_reports)
        heldout_summary = _aggregate_reports("heldout targets", diagnostic.heldout_reports)
        final_loss = float(diagnostic.losses[-1]) if diagnostic.losses else float("nan")
        rows.append(
            HamiltonianTokenDataScaleRow(
                n_train_targets=count,
                n_solution_stacks=int(dataset.stacks.shape[0]),
                final_loss=final_loss,
                train_mean_best=train_summary.mean_best,
                heldout_mean_best=heldout_summary.mean_best,
                generated_mean_best=generated_baseline.mean_best,
                heldout_delta_vs_generated=heldout_summary.mean_best - generated_baseline.mean_best,
                heldout_median_best=heldout_summary.median_best,
                heldout_min_best=heldout_summary.min_best,
                heldout_max_best=heldout_summary.max_best,
                heldout_success_95=heldout_summary.success_95,
                heldout_success_98=heldout_summary.success_98,
                heldout_success_99=heldout_summary.success_99,
            )
        )

    return HamiltonianTokenDataScaleResult(
        heldout_targets=heldout_targets,
        heldout_baseline=heldout_baseline,
        diagnostics=diagnostics,
        rows=rows,
    )


def run_hamiltonian_token_training_budget_benchmark(
    train_target_count: int,
    train_step_counts: tuple[int, ...] | list[int],
    heldout_targets: list[HamiltonianTarget],
    clifford_gates: torch.Tensor,
    clifford_labels: list[str],
    generated_gates: torch.Tensor,
    generated_labels: list[str],
    config: CircuitExperimentConfig | str,
    terms: tuple[str, ...] = ("XI", "IZ", "XX", "ZZ"),
    coefficient_scale: float = 0.35,
    time: float = 0.8,
    train_seed: int = 2607,
    dataset_seed: int = 2707,
    heldout_baseline_seed: int = 2807,
    perturb_scale: float = 0.12,
    entangler: str = "cz",
    n_entanglers: int = 2,
    n_random_candidates: int = 25_000,
    n_analytic_gates: int = 1024,
    n_haar_gates: int = 1024,
    top_k: int = 5,
    refinement_steps: int = 160,
    refinement_lr: float = 0.05,
    fidelity_threshold: float = 0.0,
    solutions_per_target: int = 2,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> HamiltonianTokenTrainingBudgetResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if train_target_count <= 0:
        raise ValueError("train_target_count must be positive")
    if not train_step_counts:
        raise ValueError("train_step_counts must contain at least one step count")
    step_counts = tuple(sorted({int(steps) for steps in train_step_counts}))
    if step_counts[0] <= 0:
        raise ValueError("train step counts must be positive")
    if not heldout_targets:
        raise ValueError("heldout_targets must contain at least one target")

    device = torch.device(device) if device is not None else generated_gates.device
    train_targets = make_random_pauli_hamiltonian_targets(
        n_targets=train_target_count,
        terms=terms,
        coefficient_scale=coefficient_scale,
        time=time,
        seed=train_seed,
        device=device,
    )
    train_dataset = generate_hamiltonian_solution_dataset(
        train_targets,
        clifford_gates=clifford_gates,
        clifford_labels=clifford_labels,
        generated_gates=generated_gates,
        generated_labels=generated_labels,
        perturb_scale=perturb_scale,
        entangler=entangler,
        n_entanglers=n_entanglers,
        n_random_candidates=n_random_candidates,
        n_analytic_gates=n_analytic_gates,
        n_haar_gates=n_haar_gates,
        top_k=max(top_k, solutions_per_target),
        seed=dataset_seed,
        refinement_steps=refinement_steps,
        refinement_lr=refinement_lr,
        fidelity_threshold=fidelity_threshold,
        solutions_per_target=solutions_per_target,
    )
    heldout_baseline = run_hamiltonian_suite_benchmark(
        heldout_targets,
        clifford_gates=clifford_gates,
        clifford_labels=clifford_labels,
        generated_gates=generated_gates,
        generated_labels=generated_labels,
        perturb_scale=perturb_scale,
        entangler=entangler,
        n_entanglers=n_entanglers,
        n_random_candidates=n_random_candidates,
        n_analytic_gates=n_analytic_gates,
        n_haar_gates=n_haar_gates,
        top_k=top_k,
        seed=heldout_baseline_seed,
        keep_fidelities=False,
    )
    generated_baseline = _aggregate_reports(
        "generated random",
        [item.generated_report for item in heldout_baseline.benchmarks],
    )

    diagnostics: dict[int, HamiltonianConditionedOverfitDiagnosticResult] = {}
    rows: list[HamiltonianTokenTrainingBudgetRow] = []
    for steps in step_counts:
        train_config = replace(config.train, num_steps=steps)
        budget_config = replace(
            config,
            name=f"{config.name}-steps{steps}",
            train=train_config,
        )
        diagnostic = run_hamiltonian_token_conditioned_overfit_diagnostic(
            train_dataset,
            heldout_targets=heldout_targets,
            config=budget_config,
            device=device,
            show_progress=show_progress,
            entangler=entangler,
            top_k=top_k,
        )
        diagnostics[steps] = diagnostic

        train_summary = _aggregate_reports("train targets", diagnostic.train_reports)
        heldout_summary = _aggregate_reports("heldout targets", diagnostic.heldout_reports)
        final_loss = float(diagnostic.losses[-1]) if diagnostic.losses else float("nan")
        rows.append(
            HamiltonianTokenTrainingBudgetRow(
                num_steps=steps,
                hidden=train_config.hidden,
                batch_size=train_config.batch_size,
                n_train_targets=len(_unique_hamiltonian_targets(train_dataset.targets)),
                n_solution_stacks=int(train_dataset.stacks.shape[0]),
                final_loss=final_loss,
                train_mean_best=train_summary.mean_best,
                heldout_mean_best=heldout_summary.mean_best,
                generated_mean_best=generated_baseline.mean_best,
                heldout_delta_vs_generated=heldout_summary.mean_best - generated_baseline.mean_best,
                heldout_median_best=heldout_summary.median_best,
                heldout_min_best=heldout_summary.min_best,
                heldout_max_best=heldout_summary.max_best,
                heldout_success_95=heldout_summary.success_95,
                heldout_success_98=heldout_summary.success_98,
                heldout_success_99=heldout_summary.success_99,
            )
        )

    return HamiltonianTokenTrainingBudgetResult(
        train_dataset=train_dataset,
        heldout_targets=heldout_targets,
        heldout_baseline=heldout_baseline,
        diagnostics=diagnostics,
        rows=rows,
    )


def _token_template_comparison_row(
    template: str,
    n_entanglers: int,
    result: HamiltonianTokenTrainingBudgetResult,
) -> HamiltonianTokenTemplateComparisonRow:
    if not result.rows:
        raise ValueError("training budget result must contain at least one row")
    budget_row = result.rows[-1]
    n_slots = _validate_solution_stacks(result.train_dataset.stacks)
    return HamiltonianTokenTemplateComparisonRow(
        template=template,
        n_entanglers=n_entanglers,
        n_slots=n_slots,
        n_train_targets=budget_row.n_train_targets,
        n_heldout_targets=len(result.heldout_targets),
        n_solution_stacks=budget_row.n_solution_stacks,
        num_steps=budget_row.num_steps,
        final_loss=budget_row.final_loss,
        train_mean_best=budget_row.train_mean_best,
        heldout_mean_best=budget_row.heldout_mean_best,
        generated_mean_best=budget_row.generated_mean_best,
        heldout_delta_vs_generated=budget_row.heldout_delta_vs_generated,
        heldout_success_95=budget_row.heldout_success_95,
        heldout_success_98=budget_row.heldout_success_98,
        heldout_success_99=budget_row.heldout_success_99,
    )


def run_hamiltonian_token_template_comparison(
    train_target_count: int,
    train_steps: int,
    heldout_targets: list[HamiltonianTarget],
    clifford_gates: torch.Tensor,
    clifford_labels: list[str],
    generated_gates: torch.Tensor,
    generated_labels: list[str],
    config: CircuitExperimentConfig | str,
    terms: tuple[str, ...] = ("XI", "IZ", "XX", "ZZ"),
    coefficient_scale: float = 0.35,
    time: float = 0.8,
    train_seed: int = 2607,
    two_dataset_seed: int = 2707,
    three_dataset_seed: int = 3707,
    two_baseline_seed: int = 2807,
    three_baseline_seed: int = 3807,
    perturb_scale: float = 0.12,
    entangler: str = "cz",
    n_random_candidates: int = 25_000,
    n_analytic_gates: int = 1024,
    n_haar_gates: int = 1024,
    top_k: int = 5,
    refinement_steps: int = 160,
    refinement_lr: float = 0.05,
    fidelity_threshold: float = 0.0,
    solutions_per_target: int = 2,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> HamiltonianTokenTemplateComparisonResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if train_target_count <= 0:
        raise ValueError("train_target_count must be positive")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if not heldout_targets:
        raise ValueError("heldout_targets must contain at least one target")

    two_entangler = run_hamiltonian_token_training_budget_benchmark(
        train_target_count=train_target_count,
        train_step_counts=(train_steps,),
        heldout_targets=heldout_targets,
        clifford_gates=clifford_gates,
        clifford_labels=clifford_labels,
        generated_gates=generated_gates,
        generated_labels=generated_labels,
        config=replace(config, name=f"{config.name}-2cz"),
        terms=terms,
        coefficient_scale=coefficient_scale,
        time=time,
        train_seed=train_seed,
        dataset_seed=two_dataset_seed,
        heldout_baseline_seed=two_baseline_seed,
        perturb_scale=perturb_scale,
        entangler=entangler,
        n_entanglers=2,
        n_random_candidates=n_random_candidates,
        n_analytic_gates=n_analytic_gates,
        n_haar_gates=n_haar_gates,
        top_k=top_k,
        refinement_steps=refinement_steps,
        refinement_lr=refinement_lr,
        fidelity_threshold=fidelity_threshold,
        solutions_per_target=solutions_per_target,
        device=device,
        show_progress=show_progress,
    )
    three_entangler = run_hamiltonian_token_training_budget_benchmark(
        train_target_count=train_target_count,
        train_step_counts=(train_steps,),
        heldout_targets=heldout_targets,
        clifford_gates=clifford_gates,
        clifford_labels=clifford_labels,
        generated_gates=generated_gates,
        generated_labels=generated_labels,
        config=replace(config, name=f"{config.name}-3cz"),
        terms=terms,
        coefficient_scale=coefficient_scale,
        time=time,
        train_seed=train_seed,
        dataset_seed=three_dataset_seed,
        heldout_baseline_seed=three_baseline_seed,
        perturb_scale=perturb_scale,
        entangler=entangler,
        n_entanglers=3,
        n_random_candidates=n_random_candidates,
        n_analytic_gates=n_analytic_gates,
        n_haar_gates=n_haar_gates,
        top_k=top_k,
        refinement_steps=refinement_steps,
        refinement_lr=refinement_lr,
        fidelity_threshold=fidelity_threshold,
        solutions_per_target=solutions_per_target,
        device=device,
        show_progress=show_progress,
    )
    return HamiltonianTokenTemplateComparisonResult(
        two_entangler=two_entangler,
        three_entangler=three_entangler,
        rows=[
            _token_template_comparison_row("2 CZ / SU(2)^6 token", 2, two_entangler),
            _token_template_comparison_row("3 CZ / SU(2)^8 token", 3, three_entangler),
        ],
    )


def run_hamiltonian_token_repeatability_benchmark(
    n_runs: int,
    train_target_count: int,
    heldout_target_count: int,
    train_steps: int,
    clifford_gates: torch.Tensor,
    clifford_labels: list[str],
    generated_gates: torch.Tensor,
    generated_labels: list[str],
    config: CircuitExperimentConfig | str,
    terms: tuple[str, ...] = ("XI", "IZ", "XX", "ZZ"),
    coefficient_scale: float = 0.35,
    time: float = 0.8,
    seed: int = 3007,
    train_seed_stride: int = 101,
    heldout_seed_stride: int = 103,
    dataset_seed_stride: int = 107,
    baseline_seed_stride: int = 109,
    perturb_scale: float = 0.12,
    entangler: str = "cz",
    n_entanglers: int = 2,
    n_random_candidates: int = 25_000,
    n_analytic_gates: int = 1024,
    n_haar_gates: int = 1024,
    top_k: int = 5,
    refinement_steps: int = 160,
    refinement_lr: float = 0.05,
    fidelity_threshold: float = 0.0,
    solutions_per_target: int = 2,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> HamiltonianTokenRepeatabilityResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    if train_target_count <= 0:
        raise ValueError("train_target_count must be positive")
    if heldout_target_count <= 0:
        raise ValueError("heldout_target_count must be positive")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")

    device = torch.device(device) if device is not None else generated_gates.device
    budget_results: list[HamiltonianTokenTrainingBudgetResult] = []
    rows: list[HamiltonianTokenRepeatabilityRow] = []
    for run in range(n_runs):
        train_seed = seed + run * train_seed_stride
        heldout_seed = seed + 10_000 + run * heldout_seed_stride
        dataset_seed = seed + 20_000 + run * dataset_seed_stride
        baseline_seed = seed + 30_000 + run * baseline_seed_stride
        heldout_targets = make_random_pauli_hamiltonian_targets(
            n_targets=heldout_target_count,
            terms=terms,
            coefficient_scale=coefficient_scale,
            time=time,
            seed=heldout_seed,
            device=device,
        )
        budget_result = run_hamiltonian_token_training_budget_benchmark(
            train_target_count=train_target_count,
            train_step_counts=(train_steps,),
            heldout_targets=heldout_targets,
            clifford_gates=clifford_gates,
            clifford_labels=clifford_labels,
            generated_gates=generated_gates,
            generated_labels=generated_labels,
            config=replace(config, name=f"{config.name}-repeat{run:02d}"),
            terms=terms,
            coefficient_scale=coefficient_scale,
            time=time,
            train_seed=train_seed,
            dataset_seed=dataset_seed,
            heldout_baseline_seed=baseline_seed,
            perturb_scale=perturb_scale,
            entangler=entangler,
            n_entanglers=n_entanglers,
            n_random_candidates=n_random_candidates,
            n_analytic_gates=n_analytic_gates,
            n_haar_gates=n_haar_gates,
            top_k=top_k,
            refinement_steps=refinement_steps,
            refinement_lr=refinement_lr,
            fidelity_threshold=fidelity_threshold,
            solutions_per_target=solutions_per_target,
            device=device,
            show_progress=show_progress,
        )
        budget_results.append(budget_result)
        budget_row = budget_result.rows[0]
        rows.append(
            HamiltonianTokenRepeatabilityRow(
                run=run,
                train_seed=train_seed,
                heldout_seed=heldout_seed,
                dataset_seed=dataset_seed,
                baseline_seed=baseline_seed,
                num_steps=budget_row.num_steps,
                n_train_targets=budget_row.n_train_targets,
                n_heldout_targets=heldout_target_count,
                n_solution_stacks=budget_row.n_solution_stacks,
                final_loss=budget_row.final_loss,
                train_mean_best=budget_row.train_mean_best,
                heldout_mean_best=budget_row.heldout_mean_best,
                generated_mean_best=budget_row.generated_mean_best,
                heldout_delta_vs_generated=budget_row.heldout_delta_vs_generated,
                heldout_success_95=budget_row.heldout_success_95,
                heldout_success_98=budget_row.heldout_success_98,
                heldout_success_99=budget_row.heldout_success_99,
            )
        )

    return HamiltonianTokenRepeatabilityResult(
        budget_results=budget_results,
        rows=rows,
    )


def run_hamiltonian_repeatability_refinement_benchmark(
    repeatability: HamiltonianTokenRepeatabilityResult,
    generated_gates: torch.Tensor,
    entangler: str = "cz",
    refinement_steps: int = 50,
    refinement_lr: float = 0.05,
    threshold: float = 0.99,
) -> HamiltonianRepeatabilityRefinementResult:
    if not repeatability.budget_results:
        raise ValueError("repeatability result must contain at least one budget result")
    if generated_gates.shape[0] == 0:
        raise ValueError("generated_gates must contain at least one gate")
    if refinement_steps <= 0:
        raise ValueError("refinement_steps must be positive")
    if refinement_lr <= 0:
        raise ValueError("refinement_lr must be positive")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    rows: list[HamiltonianRepeatabilityRefinementRow] = []
    for run, budget_result in enumerate(repeatability.budget_results):
        if not budget_result.diagnostics:
            raise ValueError("each budget result must contain at least one diagnostic")
        diagnostic = next(iter(budget_result.diagnostics.values()))
        if len(budget_result.heldout_targets) != len(diagnostic.heldout_reports):
            raise ValueError("heldout targets and token reports must have matching lengths")
        if len(budget_result.heldout_targets) != len(budget_result.heldout_baseline.benchmarks):
            raise ValueError("heldout targets and baseline benchmarks must have matching lengths")

        for i, (target, token_report, benchmark) in enumerate(
            zip(
                budget_result.heldout_targets,
                diagnostic.heldout_reports,
                budget_result.heldout_baseline.benchmarks,
            )
        ):
            token_candidate = token_report.candidates[0]
            token_stack_index = token_candidate.slot_indices[0]
            token_stack = diagnostic.heldout_generated_by_target[i, token_stack_index]
            token_refinement = refine_two_entangler_candidate(
                token_stack,
                _candidate_from_stack(
                    target,
                    "token",
                    token_candidate.fidelity,
                    n_slots=token_stack.shape[0],
                    entangler=entangler,
                ),
                target_unitary=target.unitary,
                entangler=entangler,
                num_steps=refinement_steps,
                lr=refinement_lr,
            )
            token_movements, token_movement_mean, token_movement_max = _refinement_movement(
                token_stack,
                token_refinement.refined_gates,
            )
            rows.append(
                HamiltonianRepeatabilityRefinementRow(
                    run=run,
                    target=target.name,
                    source="token",
                    initial_fidelity=token_refinement.initial_fidelity,
                    refined_fidelity=token_refinement.refined_fidelity,
                    steps_to_threshold=_steps_to_threshold(
                        token_refinement.initial_fidelity,
                        token_refinement.fidelity_trace,
                        threshold,
                    ),
                    slot_movements=token_movements,
                    movement_mean=token_movement_mean,
                    movement_max=token_movement_max,
                )
            )

            generated_candidate = benchmark.generated_report.candidates[0]
            generated_start = q_normalize(generated_gates[list(generated_candidate.slot_indices)])
            generated_refinement = refine_two_entangler_candidate(
                generated_gates,
                generated_candidate,
                target_unitary=target.unitary,
                entangler=entangler,
                num_steps=refinement_steps,
                lr=refinement_lr,
            )
            generated_movements, generated_movement_mean, generated_movement_max = _refinement_movement(
                generated_start,
                generated_refinement.refined_gates,
            )
            rows.append(
                HamiltonianRepeatabilityRefinementRow(
                    run=run,
                    target=target.name,
                    source="generated-search",
                    initial_fidelity=generated_refinement.initial_fidelity,
                    refined_fidelity=generated_refinement.refined_fidelity,
                    steps_to_threshold=_steps_to_threshold(
                        generated_refinement.initial_fidelity,
                        generated_refinement.fidelity_trace,
                        threshold,
                    ),
                    slot_movements=generated_movements,
                    movement_mean=generated_movement_mean,
                    movement_max=generated_movement_max,
                )
            )

    return HamiltonianRepeatabilityRefinementResult(
        repeatability=repeatability,
        rows=rows,
        threshold=threshold,
    )


def evaluate_hamiltonian_conditioned_denoising(
    model: TargetConditionedCircuitDenoiser,
    dataset: HamiltonianSolutionDataset,
    schedule: DiffusionSchedule,
    timesteps: tuple[int, ...] | None = None,
    n_terms: int = 128,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> list[HamiltonianDenoiseDiagnosticRow]:
    if not dataset.targets:
        raise ValueError("dataset must contain at least one Hamiltonian target")
    _validate_solution_stacks(dataset.stacks)
    if dataset.stacks.shape[0] != len(dataset.targets):
        raise ValueError("dataset.stacks must contain one stack per Hamiltonian target")

    device = torch.device(device) if device is not None else next(model.parameters()).device
    timesteps = timesteps or (1, max(1, schedule.T // 4), max(1, schedule.T // 2), schedule.T)
    for timestep in timesteps:
        if timestep < 1 or timestep > schedule.T:
            raise ValueError(f"timestep {timestep} is outside [1, {schedule.T}]")

    stacks = q_normalize(dataset.stacks.to(device=device))
    features = hamiltonian_target_features(dataset.targets).to(device=device)
    if features.shape[1] != model.target_dim:
        raise ValueError(f"Model expects {model.target_dim} Hamiltonian features, got {features.shape[1]}")

    _, _, sigmas = schedule.tensors(device)
    rows = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for timestep in timesteps:
            t_idx = torch.full((stacks.shape[0],), timestep, device=device, dtype=torch.long)
            qt_stack, eps_target = circuit_forward_heat_target(
                stacks,
                t_idx,
                schedule=schedule,
                n_terms=n_terms,
            )
            eps_pred = _predict_hamiltonian_eps(model, qt_stack, t_idx, features)

            mse = F.mse_loss(eps_pred, eps_target).item()
            zero_mse = eps_target.square().mean().item()
            flat_pred = eps_pred.reshape(-1, 3)
            flat_target = eps_target.reshape(-1, 3)
            cosine = F.cosine_similarity(flat_pred, flat_target, dim=-1, eps=1e-8).mean().item()
            target_norm = flat_target.norm(dim=-1).mean().item()
            pred_norm = flat_pred.norm(dim=-1).mean().item()
            rows.append(
                HamiltonianDenoiseDiagnosticRow(
                    timestep=int(timestep),
                    sigma=float(sigmas[timestep - 1].item()),
                    mse=float(mse),
                    zero_mse=float(zero_mse),
                    relative_mse=float(mse / max(zero_mse, 1e-12)),
                    cosine=float(cosine),
                    target_norm=float(target_norm),
                    pred_norm=float(pred_norm),
                )
            )
    return rows


def evaluate_hamiltonian_skeleton_conditioned_denoising(
    model: TargetLabelConditionedCircuitDenoiser,
    dataset: HamiltonianSolutionDataset,
    label_names: tuple[str, ...] | list[str],
    schedule: DiffusionSchedule,
    timesteps: tuple[int, ...] | None = None,
    n_terms: int = 128,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> list[HamiltonianDenoiseDiagnosticRow]:
    if not dataset.targets:
        raise ValueError("dataset must contain at least one Hamiltonian target")
    _validate_solution_stacks(dataset.stacks)
    if dataset.stacks.shape[0] != len(dataset.targets):
        raise ValueError("dataset.stacks must contain one stack per Hamiltonian target")

    device = torch.device(device) if device is not None else next(model.parameters()).device
    timesteps = timesteps or (1, max(1, schedule.T // 4), max(1, schedule.T // 2), schedule.T)
    for timestep in timesteps:
        if timestep < 1 or timestep > schedule.T:
            raise ValueError(f"timestep {timestep} is outside [1, {schedule.T}]")

    stacks = q_normalize(dataset.stacks.to(device=device))
    features = hamiltonian_target_features(dataset.targets).to(device=device)
    if features.shape[1] != model.target_dim:
        raise ValueError(f"Model expects {model.target_dim} Hamiltonian features, got {features.shape[1]}")
    slot_labels = _slot_label_targets(dataset, label_names, device=device)
    if int(slot_labels.max().item()) >= model.num_labels or int(slot_labels.min().item()) < 0:
        raise ValueError("slot labels are outside model label range")

    _, _, sigmas = schedule.tensors(device)
    rows = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for timestep in timesteps:
            t_idx = torch.full((stacks.shape[0],), timestep, device=device, dtype=torch.long)
            qt_stack, eps_target = circuit_forward_heat_target(
                stacks,
                t_idx,
                schedule=schedule,
                n_terms=n_terms,
            )
            eps_pred = _predict_hamiltonian_skeleton_eps(model, qt_stack, t_idx, features, slot_labels)

            mse = F.mse_loss(eps_pred, eps_target).item()
            zero_mse = eps_target.square().mean().item()
            flat_pred = eps_pred.reshape(-1, 3)
            flat_target = eps_target.reshape(-1, 3)
            cosine = F.cosine_similarity(flat_pred, flat_target, dim=-1, eps=1e-8).mean().item()
            target_norm = flat_target.norm(dim=-1).mean().item()
            pred_norm = flat_pred.norm(dim=-1).mean().item()
            rows.append(
                HamiltonianDenoiseDiagnosticRow(
                    timestep=int(timestep),
                    sigma=float(sigmas[timestep - 1].item()),
                    mse=float(mse),
                    zero_mse=float(zero_mse),
                    relative_mse=float(mse / max(zero_mse, 1e-12)),
                    cosine=float(cosine),
                    target_norm=float(target_norm),
                    pred_norm=float(pred_norm),
                )
            )
    return rows


def run_hamiltonian_conditioned_denoise_diagnostic(
    train_dataset: HamiltonianSolutionDataset,
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    timesteps: tuple[int, ...] | None = None,
    seed: int = 0,
    target_scale: float = 1.0,
) -> HamiltonianDenoiseDiagnosticResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    device = torch.device(device) if device is not None else train_dataset.stacks.device
    model, losses = train_hamiltonian_conditioned_circuit_diffusion(
        train_dataset,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
        target_scale=target_scale,
    )
    rows = evaluate_hamiltonian_conditioned_denoising(
        model,
        train_dataset,
        config.schedule,
        timesteps=timesteps,
        n_terms=config.train.n_terms,
        device=device,
        seed=seed,
    )
    return HamiltonianDenoiseDiagnosticResult(
        config=config,
        model=model,
        losses=losses,
        train_dataset=train_dataset,
        rows=rows,
    )


def run_hamiltonian_skeleton_denoise_diagnostic(
    train_dataset: HamiltonianSolutionDataset,
    label_names: tuple[str, ...] | list[str],
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    timesteps: tuple[int, ...] | None = None,
    seed: int = 0,
    target_scale: float = 1.0,
) -> HamiltonianSkeletonDenoiseDiagnosticResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    device = torch.device(device) if device is not None else train_dataset.stacks.device
    label_names = tuple(label_names)
    model, losses = train_hamiltonian_skeleton_conditioned_circuit_diffusion(
        train_dataset,
        label_names,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
        target_scale=target_scale,
    )
    rows = evaluate_hamiltonian_skeleton_conditioned_denoising(
        model,
        train_dataset,
        label_names,
        config.schedule,
        timesteps=timesteps,
        n_terms=config.train.n_terms,
        device=device,
        seed=seed,
    )
    return HamiltonianSkeletonDenoiseDiagnosticResult(
        config=config,
        model=model,
        losses=losses,
        train_dataset=train_dataset,
        label_names=label_names,
        rows=rows,
    )


def run_hamiltonian_slotwise_denoise_diagnostic(
    train_dataset: HamiltonianSolutionDataset,
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    timesteps: tuple[int, ...] | None = None,
    seed: int = 0,
    target_scale: float = 1.0,
) -> HamiltonianDenoiseDiagnosticResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    device = torch.device(device) if device is not None else train_dataset.stacks.device
    model, losses = train_hamiltonian_slotwise_circuit_diffusion(
        train_dataset,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
        target_scale=target_scale,
    )
    rows = evaluate_hamiltonian_conditioned_denoising(
        model,
        train_dataset,
        config.schedule,
        timesteps=timesteps,
        n_terms=config.train.n_terms,
        device=device,
        seed=seed,
    )
    return HamiltonianDenoiseDiagnosticResult(
        config=config,
        model=model,
        losses=losses,
        train_dataset=train_dataset,
        rows=rows,
    )


def run_hamiltonian_token_denoise_diagnostic(
    train_dataset: HamiltonianSolutionDataset,
    config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    timesteps: tuple[int, ...] | None = None,
    seed: int = 0,
    target_scale: float = 1.0,
) -> HamiltonianDenoiseDiagnosticResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    device = torch.device(device) if device is not None else train_dataset.stacks.device
    model, losses = train_hamiltonian_token_circuit_diffusion(
        train_dataset,
        train_config=config.train,
        schedule=config.schedule,
        device=device,
        show_progress=show_progress,
        target_scale=target_scale,
    )
    rows = evaluate_hamiltonian_conditioned_denoising(
        model,
        train_dataset,
        config.schedule,
        timesteps=timesteps,
        n_terms=config.train.n_terms,
        device=device,
        seed=seed,
    )
    return HamiltonianDenoiseDiagnosticResult(
        config=config,
        model=model,
        losses=losses,
        train_dataset=train_dataset,
        rows=rows,
    )


def hamiltonian_denoise_diagnostic_from_model(
    model: TargetConditionedCircuitDenoiser,
    train_dataset: HamiltonianSolutionDataset,
    config: CircuitExperimentConfig | str,
    losses: list[float] | None = None,
    device: torch.device | str | None = None,
    timesteps: tuple[int, ...] | None = None,
    seed: int = 0,
) -> HamiltonianDenoiseDiagnosticResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(config, str):
        config = get_circuit_experiment_config(config)
    rows = evaluate_hamiltonian_conditioned_denoising(
        model,
        train_dataset,
        config.schedule,
        timesteps=timesteps,
        n_terms=config.train.n_terms,
        device=device,
        seed=seed,
    )
    return HamiltonianDenoiseDiagnosticResult(
        config=config,
        model=model,
        losses=list(losses or []),
        train_dataset=train_dataset,
        rows=rows,
    )


def _default_hamiltonian_denoise_ablation_configs(
    base_config: CircuitExperimentConfig,
) -> list[CircuitExperimentConfig]:
    base_train = base_config.train
    longer_steps = max(base_train.num_steps * 10, base_train.num_steps + 100)
    wider_hidden = max(base_train.hidden * 4, base_train.hidden + 128)
    variants = [
        ("current", base_train),
        ("longer", replace(base_train, num_steps=longer_steps)),
        ("wider", replace(base_train, hidden=wider_hidden)),
        ("wider-longer", replace(base_train, hidden=wider_hidden, num_steps=longer_steps)),
    ]
    return [
        replace(
            base_config,
            name=f"{base_config.name}-{name}",
            train=train_config,
        )
        for name, train_config in variants
    ]


def _hamiltonian_denoise_ablation_row(
    result: HamiltonianDenoiseDiagnosticResult,
) -> HamiltonianDenoiseAblationRow:
    if not result.rows:
        raise ValueError("diagnostic result must contain at least one denoising row")
    first = min(result.rows, key=lambda row: row.timestep)
    final = max(result.rows, key=lambda row: row.timestep)
    final_loss = float(result.losses[-1]) if result.losses else float("nan")
    return HamiltonianDenoiseAblationRow(
        name=result.config.name,
        num_steps=result.config.train.num_steps,
        hidden=result.config.train.hidden,
        final_loss=final_loss,
        t1_relative_mse=first.relative_mse,
        final_relative_mse=final.relative_mse,
        final_cosine=final.cosine,
        final_pred_target_norm_ratio=final.pred_norm / max(final.target_norm, 1e-12),
    )


def _hamiltonian_denoise_normalization_row(
    variant: str,
    result: HamiltonianDenoiseDiagnosticResult,
) -> HamiltonianDenoiseNormalizationRow:
    if not result.rows:
        raise ValueError("diagnostic result must contain at least one denoising row")
    final = max(result.rows, key=lambda row: row.timestep)
    final_loss = float(result.losses[-1]) if result.losses else float("nan")
    return HamiltonianDenoiseNormalizationRow(
        variant=variant,
        target_scale=_eps_output_scale(result.model),
        final_loss=final_loss,
        final_relative_mse=final.relative_mse,
        final_cosine=final.cosine,
        final_pred_target_norm_ratio=final.pred_norm / max(final.target_norm, 1e-12),
    )


def _hamiltonian_skeleton_denoise_comparison_row(
    variant: str,
    result: HamiltonianDenoiseDiagnosticResult | HamiltonianSkeletonDenoiseDiagnosticResult,
) -> HamiltonianSkeletonDenoiseComparisonRow:
    if not result.rows:
        raise ValueError("diagnostic result must contain at least one denoising row")
    final = max(result.rows, key=lambda row: row.timestep)
    final_loss = float(result.losses[-1]) if result.losses else float("nan")
    return HamiltonianSkeletonDenoiseComparisonRow(
        variant=variant,
        final_loss=final_loss,
        final_relative_mse=final.relative_mse,
        final_cosine=final.cosine,
        final_pred_target_norm_ratio=final.pred_norm / max(final.target_norm, 1e-12),
    )


def _hamiltonian_slotwise_denoise_comparison_row(
    variant: str,
    result: HamiltonianDenoiseDiagnosticResult,
) -> HamiltonianSlotwiseDenoiseComparisonRow:
    if not result.rows:
        raise ValueError("diagnostic result must contain at least one denoising row")
    final = max(result.rows, key=lambda row: row.timestep)
    final_loss = float(result.losses[-1]) if result.losses else float("nan")
    return HamiltonianSlotwiseDenoiseComparisonRow(
        variant=variant,
        final_loss=final_loss,
        final_relative_mse=final.relative_mse,
        final_cosine=final.cosine,
        final_pred_target_norm_ratio=final.pred_norm / max(final.target_norm, 1e-12),
    )


def _hamiltonian_token_denoise_comparison_row(
    variant: str,
    result: HamiltonianDenoiseDiagnosticResult,
) -> HamiltonianTokenDenoiseComparisonRow:
    if not result.rows:
        raise ValueError("diagnostic result must contain at least one denoising row")
    final = max(result.rows, key=lambda row: row.timestep)
    final_loss = float(result.losses[-1]) if result.losses else float("nan")
    return HamiltonianTokenDenoiseComparisonRow(
        variant=variant,
        final_loss=final_loss,
        final_relative_mse=final.relative_mse,
        final_cosine=final.cosine,
        final_pred_target_norm_ratio=final.pred_norm / max(final.target_norm, 1e-12),
    )


def run_hamiltonian_slotwise_denoise_comparison(
    train_dataset: HamiltonianSolutionDataset,
    base_config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    timesteps: tuple[int, ...] | None = None,
    seed: int = 0,
) -> HamiltonianSlotwiseDenoiseComparisonResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(base_config, str):
        base_config = get_circuit_experiment_config(base_config)
    flat = run_hamiltonian_conditioned_denoise_diagnostic(
        train_dataset,
        config=base_config,
        device=device,
        show_progress=show_progress,
        timesteps=timesteps,
        seed=seed,
    )
    slotwise = run_hamiltonian_slotwise_denoise_diagnostic(
        train_dataset,
        config=replace(base_config, name=f"{base_config.name}-slotwise"),
        device=device,
        show_progress=show_progress,
        timesteps=timesteps,
        seed=seed + 1,
    )
    return HamiltonianSlotwiseDenoiseComparisonResult(
        flat=flat,
        slotwise=slotwise,
        rows=[
            _hamiltonian_slotwise_denoise_comparison_row("flat MLP", flat),
            _hamiltonian_slotwise_denoise_comparison_row("slot-wise MLP", slotwise),
        ],
    )


def run_hamiltonian_token_denoise_comparison(
    train_dataset: HamiltonianSolutionDataset,
    base_config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    timesteps: tuple[int, ...] | None = None,
    seed: int = 0,
) -> HamiltonianTokenDenoiseComparisonResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(base_config, str):
        base_config = get_circuit_experiment_config(base_config)
    flat = run_hamiltonian_conditioned_denoise_diagnostic(
        train_dataset,
        config=base_config,
        device=device,
        show_progress=show_progress,
        timesteps=timesteps,
        seed=seed,
    )
    token = run_hamiltonian_token_denoise_diagnostic(
        train_dataset,
        config=replace(base_config, name=f"{base_config.name}-token"),
        device=device,
        show_progress=show_progress,
        timesteps=timesteps,
        seed=seed + 1,
    )
    return HamiltonianTokenDenoiseComparisonResult(
        flat=flat,
        token=token,
        rows=[
            _hamiltonian_token_denoise_comparison_row("flat MLP", flat),
            _hamiltonian_token_denoise_comparison_row("circuit-token", token),
        ],
    )


def run_hamiltonian_skeleton_denoise_comparison(
    train_dataset: HamiltonianSolutionDataset,
    label_names: tuple[str, ...] | list[str],
    base_config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    timesteps: tuple[int, ...] | None = None,
    seed: int = 0,
) -> HamiltonianSkeletonDenoiseComparisonResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(base_config, str):
        base_config = get_circuit_experiment_config(base_config)
    unconditioned = run_hamiltonian_conditioned_denoise_diagnostic(
        train_dataset,
        config=base_config,
        device=device,
        show_progress=show_progress,
        timesteps=timesteps,
        seed=seed,
    )
    skeleton_conditioned = run_hamiltonian_skeleton_denoise_diagnostic(
        train_dataset,
        label_names=label_names,
        config=replace(base_config, name=f"{base_config.name}-skeleton"),
        device=device,
        show_progress=show_progress,
        timesteps=timesteps,
        seed=seed + 1,
    )
    return HamiltonianSkeletonDenoiseComparisonResult(
        unconditioned=unconditioned,
        skeleton_conditioned=skeleton_conditioned,
        rows=[
            _hamiltonian_skeleton_denoise_comparison_row("H-only", unconditioned),
            _hamiltonian_skeleton_denoise_comparison_row("H+slot labels", skeleton_conditioned),
        ],
    )


def run_hamiltonian_denoise_normalization_comparison(
    train_dataset: HamiltonianSolutionDataset,
    base_config: CircuitExperimentConfig | str,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    timesteps: tuple[int, ...] | None = None,
    seed: int = 0,
) -> HamiltonianDenoiseNormalizationResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(base_config, str):
        base_config = get_circuit_experiment_config(base_config)
    device = torch.device(device) if device is not None else train_dataset.stacks.device
    target_scale = estimate_hamiltonian_denoise_target_scale(
        train_dataset,
        base_config.schedule,
        batch_size=base_config.train.batch_size,
        n_batches=8,
        n_terms=base_config.train.n_terms,
        device=device,
        seed=seed,
    )
    wider_config = replace(
        base_config,
        name=f"{base_config.name}-normalized-wider",
        train=replace(base_config.train, hidden=max(base_config.train.hidden * 4, base_config.train.hidden + 128)),
    )
    variants = [
        ("unnormalized", base_config, 1.0),
        ("normalized", replace(base_config, name=f"{base_config.name}-normalized"), target_scale),
        ("normalized+wider", wider_config, target_scale),
    ]

    diagnostics = []
    rows = []
    for index, (variant, config, scale) in enumerate(variants):
        result = run_hamiltonian_conditioned_denoise_diagnostic(
            train_dataset,
            config=config,
            device=device,
            show_progress=show_progress,
            timesteps=timesteps,
            seed=seed + index + 1,
            target_scale=scale,
        )
        diagnostics.append(result)
        rows.append(_hamiltonian_denoise_normalization_row(variant, result))

    return HamiltonianDenoiseNormalizationResult(
        train_dataset=train_dataset,
        diagnostics=diagnostics,
        rows=rows,
    )


def run_hamiltonian_denoise_ablation(
    train_dataset: HamiltonianSolutionDataset,
    base_config: CircuitExperimentConfig | str,
    configs: list[CircuitExperimentConfig] | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    timesteps: tuple[int, ...] | None = None,
    seed: int = 0,
) -> HamiltonianDenoiseAblationResult:
    from .circuit import get_circuit_experiment_config

    if isinstance(base_config, str):
        base_config = get_circuit_experiment_config(base_config)
    configs = configs or _default_hamiltonian_denoise_ablation_configs(base_config)
    diagnostics = []
    for index, config in enumerate(configs):
        diagnostics.append(
            run_hamiltonian_conditioned_denoise_diagnostic(
                train_dataset,
                config=config,
                device=device,
                show_progress=show_progress,
                timesteps=timesteps,
                seed=seed + index,
            )
        )
    rows = [_hamiltonian_denoise_ablation_row(item) for item in diagnostics]
    return HamiltonianDenoiseAblationResult(
        train_dataset=train_dataset,
        diagnostics=diagnostics,
        rows=rows,
    )


def _slot_label_targets(
    dataset: HamiltonianSolutionDataset,
    label_names: tuple[str, ...] | list[str],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    label_to_index = {label: i for i, label in enumerate(label_names)}
    rows = []
    n_slots = _validate_solution_stacks(dataset.stacks)
    for refinement in dataset.refinements:
        if len(refinement.slot_labels) != n_slots:
            raise ValueError(f"Hamiltonian prior training expects {n_slots} slot labels per refinement")
        row = []
        for label in refinement.slot_labels:
            if label not in label_to_index:
                raise ValueError(f"Unknown slot label {label!r}")
            row.append(label_to_index[label])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


def train_hamiltonian_slot_prior(
    dataset: HamiltonianSolutionDataset,
    label_names: tuple[str, ...] | list[str],
    config: HamiltonianPriorTrainConfig | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> HamiltonianPriorResult:
    if not dataset.targets:
        raise ValueError("dataset must contain at least one target")
    if not label_names:
        raise ValueError("label_names must contain at least one label")
    config = config or HamiltonianPriorTrainConfig()
    device = device or dataset.stacks.device

    torch.manual_seed(config.seed)
    features = hamiltonian_target_features(dataset.targets).to(device=device, dtype=torch.float32)
    targets = _slot_label_targets(dataset, label_names, device=device)
    model = HamiltonianSlotPriorPredictor(
        input_dim=features.shape[1],
        hidden=config.hidden,
        n_slots=targets.shape[1],
        n_labels=len(label_names),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    losses = []
    iterator = range(config.num_steps)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Training Hamiltonian slot prior")
    for _ in iterator:
        logits = model(features)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if show_progress:
            iterator.set_postfix(loss=losses[-1])

    with torch.no_grad():
        predicted = model(features).argmax(dim=-1)
        accuracy = (predicted == targets).float().mean().item()
    return HamiltonianPriorResult(
        model=model,
        losses=losses,
        label_names=tuple(label_names),
        train_accuracy=accuracy,
    )


def _indices_by_label(local_labels: list[str], label_names: tuple[str, ...], device: torch.device) -> list[torch.Tensor]:
    pools = []
    for label in label_names:
        indices = [i for i, local_label in enumerate(local_labels) if local_label == label]
        if not indices:
            raise ValueError(f"No local gates found for label {label!r}")
        pools.append(torch.tensor(indices, dtype=torch.long, device=device))
    return pools


def _sample_prior_slots(
    probabilities: torch.Tensor,
    local_labels: list[str],
    label_names: tuple[str, ...],
    n_candidates: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(label_names):
        raise ValueError("probabilities must have shape (n_slots, n_labels)")
    device = probabilities.device
    n_slots = probabilities.shape[0]
    pools = _indices_by_label(local_labels, label_names, device=device)
    label_slots = torch.multinomial(probabilities, num_samples=n_candidates, replacement=True, generator=generator).T
    slots = torch.empty(n_candidates, n_slots, dtype=torch.long, device=device)
    for slot in range(n_slots):
        for label_index, pool in enumerate(pools):
            mask = label_slots[:, slot] == label_index
            n_items = int(mask.sum().item())
            if n_items == 0:
                continue
            choices = torch.randint(pool.numel(), (n_items,), device=device, generator=generator)
            slots[mask, slot] = pool[choices]
    return slots


def synthesize_hamiltonian_prior_search_report(
    model: HamiltonianSlotPriorPredictor,
    target: HamiltonianTarget,
    local_gates: torch.Tensor,
    local_labels: list[str],
    label_names: tuple[str, ...] | list[str],
    entangler: str = "cz",
    n_candidates: int = 100_000,
    top_k: int = 5,
    seed: int = 0,
    name: str | None = None,
    keep_fidelities: bool = False,
    prior_weight: float = 1.0,
) -> SynthesisReport:
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if local_gates.shape[0] == 0:
        raise ValueError("local_gates must contain at least one gate")
    if not (0.0 <= prior_weight <= 1.0):
        raise ValueError("prior_weight must be between 0 and 1")
    label_names = tuple(label_names)
    device = local_gates.device
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        features = hamiltonian_target_features([target]).to(device=device, dtype=torch.float32)
        probabilities = F.softmax(model(features)[0], dim=-1)
        uniform = torch.full_like(probabilities, 1.0 / probabilities.shape[-1])
        probabilities = prior_weight * probabilities + (1.0 - prior_weight) * uniform

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    indices = _sample_prior_slots(
        probabilities=probabilities,
        local_labels=local_labels,
        label_names=label_names,
        n_candidates=n_candidates,
        generator=generator,
    )

    entangler = entangler.lower()
    units = quaternion_to_unitary(local_gates)
    target_unitary = target.unitary.to(device=device, dtype=torch.complex64)
    entangler_unitary = two_qubit_gate(entangler, device=device)
    n_slots = indices.shape[1]
    n_entanglers = n_slots // 2 - 1
    unitaries = compose_local_entangler_chain_units(units[indices], entangler_unitary)
    fidelities = unitary_fidelity_batch(unitaries, target_unitary)
    values, rows = torch.topk(fidelities, k=min(top_k, fidelities.numel()))

    candidates = []
    for value, row in zip(values.tolist(), rows.tolist()):
        slots = indices[row].tolist()
        candidates.append(
            SynthesisCandidate(
                target=target.name.lower(),
                template=f"hamiltonian-prior-{n_entanglers}-entangler-local",
                entangler=entangler,
                fidelity=value,
                slot_indices=tuple(slots),
                slot_labels=tuple(local_labels[index] for index in slots),
            )
        )
    return SynthesisReport(
        name=name or f"{target.name} Hamiltonian prior search",
        mode=f"prior alpha={prior_weight:g}",
        target=target.name.lower(),
        entangler=entangler,
        candidates=candidates,
        fidelities=fidelities.tolist() if keep_fidelities else tuple(float(candidate.fidelity) for candidate in candidates),
    )


def run_hamiltonian_prior_search_benchmark(
    prior: HamiltonianPriorResult,
    targets: list[HamiltonianTarget],
    local_gates: torch.Tensor,
    local_labels: list[str],
    entangler: str = "cz",
    n_candidates: int = 100_000,
    top_k: int = 5,
    seed: int = 0,
    keep_fidelities: bool = False,
    prior_weight: float = 1.0,
) -> HamiltonianPriorSearchResult:
    if not targets:
        raise ValueError("targets must contain at least one target")
    benchmarks = []
    for i, target in enumerate(targets):
        uniform_report = synthesize_unitary_two_entangler_random_report(
            local_gates,
            target_unitary=target.unitary,
            target_name=target.name,
            entangler=entangler,
            n_candidates=n_candidates,
            top_k=top_k,
            local_labels=local_labels,
            seed=seed + 10_000 + i,
            name=f"{target.name} uniform generated search",
            keep_fidelities=keep_fidelities,
        )
        prior_report = synthesize_hamiltonian_prior_search_report(
            prior.model,
            target,
            local_gates=local_gates,
            local_labels=local_labels,
            label_names=prior.label_names,
            entangler=entangler,
            n_candidates=n_candidates,
            top_k=top_k,
            seed=seed + 20_000 + i,
            name=f"{target.name} learned-prior alpha={prior_weight:g} search",
            keep_fidelities=keep_fidelities,
            prior_weight=prior_weight,
        )
        benchmarks.append(
            HamiltonianPriorSearchBenchmark(
                target=target,
                uniform_report=uniform_report,
                prior_report=prior_report,
            )
        )
    return HamiltonianPriorSearchResult(benchmarks=benchmarks)


def run_hamiltonian_prior_mixture_sweep(
    prior: HamiltonianPriorResult,
    targets: list[HamiltonianTarget],
    local_gates: torch.Tensor,
    local_labels: list[str],
    alphas: tuple[float, ...] | list[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    entangler: str = "cz",
    n_candidates: int = 100_000,
    top_k: int = 5,
    seed: int = 0,
    keep_fidelities: bool = False,
) -> HamiltonianPriorMixtureResult:
    if not alphas:
        raise ValueError("alphas must contain at least one value")
    alpha_results = {}
    for i, alpha in enumerate(alphas):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alphas must be between 0 and 1")
        alpha_results[float(alpha)] = run_hamiltonian_prior_search_benchmark(
            prior,
            targets,
            local_gates=local_gates,
            local_labels=local_labels,
            entangler=entangler,
            n_candidates=n_candidates,
            top_k=top_k,
            seed=seed + 50_000 * i,
            keep_fidelities=keep_fidelities,
            prior_weight=float(alpha),
        )
    return HamiltonianPriorMixtureResult(alpha_results=alpha_results)


def refine_hamiltonian_prior_mixture(
    result: HamiltonianPriorMixtureResult,
    local_gates: torch.Tensor,
    entangler: str = "cz",
    refinement_steps: int = 100,
    refinement_lr: float = 0.05,
    threshold: float = 0.99,
) -> HamiltonianMixtureRefinementResult:
    if not result.alpha_results:
        raise ValueError("result must contain at least one alpha")
    if local_gates.shape[0] == 0:
        raise ValueError("local_gates must contain at least one gate")
    if refinement_steps <= 0:
        raise ValueError("refinement_steps must be positive")
    if refinement_lr <= 0:
        raise ValueError("refinement_lr must be positive")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    rows = []
    for alpha, alpha_result in result.alpha_results.items():
        for benchmark in alpha_result.benchmarks:
            candidate = benchmark.prior_report.candidates[0]
            refinement = refine_two_entangler_candidate(
                local_gates,
                candidate,
                target_unitary=benchmark.target.unitary,
                entangler=entangler,
                num_steps=refinement_steps,
                lr=refinement_lr,
            )
            rows.append(
                HamiltonianMixtureRefinementRow(
                    target=benchmark.target.name,
                    alpha=float(alpha),
                    initial_fidelity=refinement.initial_fidelity,
                    refined_fidelity=refinement.refined_fidelity,
                    steps_to_threshold=_steps_to_threshold(
                        refinement.initial_fidelity,
                        refinement.fidelity_trace,
                        threshold,
                    ),
                )
            )
    return HamiltonianMixtureRefinementResult(rows=rows, threshold=threshold)


def refine_hamiltonian_prior_mixture_budget_sweep(
    result: HamiltonianPriorMixtureResult,
    local_gates: torch.Tensor,
    budgets: tuple[int, ...] | list[int] = (5, 10, 20, 50),
    entangler: str = "cz",
    refinement_lr: float = 0.05,
    threshold: float = 0.99,
) -> HamiltonianBudgetRefinementResult:
    if not budgets:
        raise ValueError("budgets must contain at least one value")
    rows = []
    for budget in budgets:
        if budget <= 0:
            raise ValueError("budgets must be positive")
        refined = refine_hamiltonian_prior_mixture(
            result,
            local_gates=local_gates,
            entangler=entangler,
            refinement_steps=int(budget),
            refinement_lr=refinement_lr,
            threshold=threshold,
        )
        for row in refined.rows:
            rows.append(
                HamiltonianBudgetRefinementRow(
                    budget=int(budget),
                    target=row.target,
                    alpha=row.alpha,
                    initial_fidelity=row.initial_fidelity,
                    refined_fidelity=row.refined_fidelity,
                    reached_threshold=row.refined_fidelity >= threshold,
                )
            )
    return HamiltonianBudgetRefinementResult(rows=rows, threshold=threshold)


def run_hamiltonian_seed_ablation(
    targets: list[HamiltonianTarget],
    predicted_stacks: torch.Tensor,
    generated_suite: HamiltonianSuiteResult,
    clifford_gates: torch.Tensor,
    generated_gates: torch.Tensor,
    entangler: str = "cz",
    n_haar_seeds: int = 1,
    refinement_steps: int = 100,
    refinement_lr: float = 0.05,
    threshold: float = 0.99,
    seed: int = 0,
) -> HamiltonianSeedAblationResult:
    if not targets:
        raise ValueError("targets must contain at least one target")
    if len(targets) != len(generated_suite.benchmarks):
        raise ValueError("targets and generated_suite must have the same length")
    n_slots = _validate_solution_stacks(predicted_stacks, name="predicted_stacks")
    if predicted_stacks.shape[0] != len(targets):
        raise ValueError(f"predicted_stacks must have shape (n_targets, {n_slots}, 4)")
    if n_haar_seeds <= 0:
        raise ValueError("n_haar_seeds must be positive")
    if refinement_steps <= 0:
        raise ValueError("refinement_steps must be positive")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    device = predicted_stacks.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    rows: list[HamiltonianSeedAblationRow] = []

    def add_seed(target: HamiltonianTarget, seed_type: str, stack: torch.Tensor) -> None:
        initial = _stack_fidelity(stack, target, entangler=entangler)
        candidate = _candidate_from_stack(target, seed_type, initial, n_slots=stack.shape[0], entangler=entangler)
        refinement = refine_two_entangler_candidate(
            stack,
            candidate,
            target_unitary=target.unitary,
            entangler=entangler,
            num_steps=refinement_steps,
            lr=refinement_lr,
        )
        rows.append(
            HamiltonianSeedAblationRow(
                target=target.name,
                seed_type=seed_type,
                initial_fidelity=initial,
                refined_fidelity=refinement.refined_fidelity,
                steps_to_threshold=_steps_to_threshold(initial, refinement.fidelity_trace, threshold),
            )
        )

    for i, (target, benchmark) in enumerate(zip(targets, generated_suite.benchmarks)):
        add_seed(target, "mlp", predicted_stacks[i])
        generated_candidate = benchmark.generated_report.candidates[0]
        add_seed(target, "generated-search", generated_gates[list(generated_candidate.slot_indices)])
        clifford_candidate = benchmark.clifford_report.candidates[0]
        add_seed(target, "clifford-search", clifford_gates[list(clifford_candidate.slot_indices)])

        best_haar_stack = None
        best_haar_fidelity = -1.0
        for _ in range(n_haar_seeds):
            stack = sample_haar(n_slots, device=device, generator=generator)
            fidelity = _stack_fidelity(stack, target, entangler=entangler)
            if fidelity > best_haar_fidelity:
                best_haar_stack = stack
                best_haar_fidelity = fidelity
        if best_haar_stack is None:
            raise RuntimeError("failed to sample Haar seed")
        add_seed(target, "haar", best_haar_stack)

    return HamiltonianSeedAblationResult(rows=rows, threshold=threshold)


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


def summarize_hamiltonian_conditioned_diffusion(
    baseline: HamiltonianSuiteResult,
    conditioned: HamiltonianConditionedDiffusionResult,
) -> list[HiddenShallowCircuitAggregate]:
    if len(baseline.benchmarks) != len(conditioned.reports):
        raise ValueError("baseline and conditioned result must cover the same number of targets")
    return [
        *summarize_hamiltonian_suite(baseline),
        _aggregate_reports("Hamiltonian-conditioned diffusion", conditioned.reports),
    ]


def summarize_hamiltonian_token_conditioned_diffusion(
    baseline: HamiltonianSuiteResult,
    conditioned: HamiltonianConditionedDiffusionResult,
) -> list[HiddenShallowCircuitAggregate]:
    if len(baseline.benchmarks) != len(conditioned.reports):
        raise ValueError("baseline and conditioned result must cover the same number of targets")
    return [
        *summarize_hamiltonian_suite(baseline),
        _aggregate_reports("Hamiltonian circuit-token diffusion", conditioned.reports),
    ]


def summarize_hamiltonian_token_heldout_comparison(
    heldout_baseline: HamiltonianSuiteResult,
    token_diagnostic: HamiltonianConditionedOverfitDiagnosticResult,
) -> list[HiddenShallowCircuitAggregate]:
    if len(heldout_baseline.benchmarks) != len(token_diagnostic.heldout_reports):
        raise ValueError("heldout baseline and token diagnostic must cover the same number of targets")
    return [
        *summarize_hamiltonian_suite(heldout_baseline),
        _aggregate_reports("Hamiltonian circuit-token heldout", token_diagnostic.heldout_reports),
    ]


def summarize_hamiltonian_token_data_scale(
    result: HamiltonianTokenDataScaleResult,
) -> list[HamiltonianTokenDataScaleRow]:
    return result.rows


def summarize_hamiltonian_token_training_budget(
    result: HamiltonianTokenTrainingBudgetResult,
) -> list[HamiltonianTokenTrainingBudgetRow]:
    return result.rows


def summarize_hamiltonian_token_template_comparison(
    result: HamiltonianTokenTemplateComparisonResult,
) -> list[HamiltonianTokenTemplateComparisonRow]:
    return result.rows


def summarize_hamiltonian_token_repeatability(
    result: HamiltonianTokenRepeatabilityResult,
) -> list[HamiltonianTokenRepeatabilityRow]:
    return result.rows


def summarize_hamiltonian_repeatability_refinement(
    result: HamiltonianRepeatabilityRefinementResult,
) -> list[HamiltonianRepeatabilityRefinementRow]:
    return result.rows


def summarize_hamiltonian_level1_headline(
    result: HamiltonianRepeatabilityRefinementResult,
) -> HamiltonianLevel1HeadlineResult:
    repeatability_rows = summarize_hamiltonian_token_repeatability(result.repeatability)
    refinement_groups = _repeatability_refinement_groups(result)
    if not repeatability_rows:
        raise ValueError("repeatability result must contain at least one row")
    if "token" not in refinement_groups or "generated-search" not in refinement_groups:
        raise ValueError("refinement result must contain token and generated-search rows")

    def source_row(
        source: str,
        proposal_values: list[float],
    ) -> HamiltonianLevel1HeadlineRow:
        rows = refinement_groups[source]
        after = torch.tensor([row.refined_fidelity for row in rows], dtype=torch.float32)
        reached = after >= result.threshold
        steps = torch.tensor(
            [row.steps_to_threshold for row in rows if row.steps_to_threshold >= 0],
            dtype=torch.float32,
        )
        mean_moves = torch.tensor([row.movement_mean for row in rows], dtype=torch.float32)
        max_moves = torch.tensor([row.movement_max for row in rows], dtype=torch.float32)
        proposal_mean, proposal_run_std = _mean_std(proposal_values)
        return HamiltonianLevel1HeadlineRow(
            source=source,
            n_targets=len(rows),
            proposal_mean=proposal_mean,
            proposal_run_std=proposal_run_std,
            refined_mean=float(after.mean().item()),
            refinement_success=float(reached.float().mean().item()),
            median_steps=float(steps.median().item()) if steps.numel() else float("nan"),
            mean_movement=float(mean_moves.mean().item()),
            max_movement=float(max_moves.max().item()),
        )

    advantage_mean, advantage_std = _mean_std([row.heldout_delta_vs_generated for row in repeatability_rows])
    return HamiltonianLevel1HeadlineResult(
        rows=[
            source_row("token", [row.heldout_mean_best for row in repeatability_rows]),
            source_row("generated-search", [row.generated_mean_best for row in repeatability_rows]),
        ],
        n_runs=len(repeatability_rows),
        threshold=result.threshold,
        proposal_advantage_mean=advantage_mean,
        proposal_advantage_std=advantage_std,
    )


def summarize_hamiltonian_conditioned_overfit_diagnostic(
    result: HamiltonianConditionedOverfitDiagnosticResult,
) -> list[HiddenShallowCircuitAggregate]:
    return [
        _aggregate_reports("train targets", result.train_reports),
        _aggregate_reports("heldout targets", result.heldout_reports),
    ]


def summarize_hamiltonian_denoise_diagnostic(
    result: HamiltonianDenoiseDiagnosticResult,
) -> list[HamiltonianDenoiseDiagnosticRow]:
    return result.rows


def summarize_hamiltonian_denoise_ablation(
    result: HamiltonianDenoiseAblationResult,
) -> list[HamiltonianDenoiseAblationRow]:
    return result.rows


def summarize_hamiltonian_denoise_normalization(
    result: HamiltonianDenoiseNormalizationResult,
) -> list[HamiltonianDenoiseNormalizationRow]:
    return result.rows


def summarize_hamiltonian_skeleton_denoise_comparison(
    result: HamiltonianSkeletonDenoiseComparisonResult,
) -> list[HamiltonianSkeletonDenoiseComparisonRow]:
    return result.rows


def summarize_hamiltonian_slotwise_denoise_comparison(
    result: HamiltonianSlotwiseDenoiseComparisonResult,
) -> list[HamiltonianSlotwiseDenoiseComparisonRow]:
    return result.rows


def summarize_hamiltonian_token_denoise_comparison(
    result: HamiltonianTokenDenoiseComparisonResult,
) -> list[HamiltonianTokenDenoiseComparisonRow]:
    return result.rows


def summarize_hamiltonian_prior_search(result: HamiltonianPriorSearchResult) -> list[HiddenShallowCircuitAggregate]:
    if not result.benchmarks:
        raise ValueError("result must contain at least one benchmark")
    return [
        _aggregate_reports("uniform generated", [item.uniform_report for item in result.benchmarks]),
        _aggregate_reports("learned prior", [item.prior_report for item in result.benchmarks]),
    ]


def summarize_hamiltonian_prior_mixture(result: HamiltonianPriorMixtureResult) -> list[HiddenShallowCircuitAggregate]:
    rows = []
    for alpha, alpha_result in result.alpha_results.items():
        rows.append(_aggregate_reports(f"alpha={alpha:g}", [item.prior_report for item in alpha_result.benchmarks]))
    return rows


def _mixture_refinement_groups(
    result: HamiltonianMixtureRefinementResult,
) -> dict[float, list[HamiltonianMixtureRefinementRow]]:
    groups: dict[float, list[HamiltonianMixtureRefinementRow]] = {}
    for row in result.rows:
        groups.setdefault(row.alpha, []).append(row)
    return groups


def _budget_refinement_groups(
    result: HamiltonianBudgetRefinementResult,
) -> dict[tuple[int, float], list[HamiltonianBudgetRefinementRow]]:
    groups: dict[tuple[int, float], list[HamiltonianBudgetRefinementRow]] = {}
    for row in result.rows:
        groups.setdefault((row.budget, row.alpha), []).append(row)
    return groups


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


def print_hamiltonian_conditioned_diffusion(
    result: HamiltonianConditionedDiffusionResult,
    max_rows: int | None = 6,
) -> None:
    header = "target      conditioned   best labels"
    print(header)
    print("-" * len(header))
    rows = list(zip(result.eval_targets, result.reports))
    rows = rows if max_rows is None else rows[:max_rows]
    for target, report in rows:
        labels = ", ".join(label if label is not None else "?" for label in report.candidates[0].slot_labels)
        print(f"{target.name:<11} {_best(report):>11.4f}   {labels}")
    if max_rows is not None and len(result.reports) > max_rows:
        print(f"... {len(result.reports) - max_rows} more")


def print_hamiltonian_conditioned_diffusion_summary(
    baseline: HamiltonianSuiteResult,
    conditioned: HamiltonianConditionedDiffusionResult,
) -> None:
    header = "mode                              n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_hamiltonian_conditioned_diffusion(baseline, conditioned):
        print(
            f"{item.mode:<33} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_hamiltonian_conditioned_overfit_diagnostic(
    result: HamiltonianConditionedOverfitDiagnosticResult,
    max_rows: int | None = 6,
) -> None:
    header = "split    target      conditioned"
    print(header)
    print("-" * len(header))
    rows = [
        *[("train", target, report) for target, report in zip(result.train_targets, result.train_reports)],
        *[("heldout", target, report) for target, report in zip(result.heldout_targets, result.heldout_reports)],
    ]
    rows = rows if max_rows is None else rows[:max_rows]
    for split, target, report in rows:
        print(f"{split:<8} {target.name:<11} {_best(report):>11.4f}")
    total_rows = len(result.train_reports) + len(result.heldout_reports)
    if max_rows is not None and total_rows > max_rows:
        print(f"... {total_rows - max_rows} more")


def print_hamiltonian_conditioned_overfit_summary(
    result: HamiltonianConditionedOverfitDiagnosticResult,
) -> None:
    header = "split            n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_hamiltonian_conditioned_overfit_diagnostic(result):
        print(
            f"{item.mode:<15} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_hamiltonian_token_conditioned_diffusion_summary(
    baseline: HamiltonianSuiteResult,
    conditioned: HamiltonianConditionedDiffusionResult,
) -> None:
    header = "mode                                      n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_hamiltonian_token_conditioned_diffusion(baseline, conditioned):
        print(
            f"{item.mode:<41} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_hamiltonian_token_heldout_comparison_summary(
    heldout_baseline: HamiltonianSuiteResult,
    token_diagnostic: HamiltonianConditionedOverfitDiagnosticResult,
) -> None:
    header = "mode                                      n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_hamiltonian_token_heldout_comparison(heldout_baseline, token_diagnostic):
        print(
            f"{item.mode:<41} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_hamiltonian_token_data_scale_summary(result: HamiltonianTokenDataScaleResult) -> None:
    header = "n train   stacks   final loss   train mean   heldout mean   gen mean   token-gen   >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for row in summarize_hamiltonian_token_data_scale(result):
        print(
            f"{row.n_train_targets:<9} "
            f"{row.n_solution_stacks:<6} "
            f"{row.final_loss:>10.6f}   "
            f"{row.train_mean_best:>10.4f}   "
            f"{row.heldout_mean_best:>12.4f}   "
            f"{row.generated_mean_best:>8.4f}   "
            f"{row.heldout_delta_vs_generated:>+9.4f}   "
            f"{row.heldout_success_95:>6.1%}   {row.heldout_success_98:>6.1%}   {row.heldout_success_99:>6.1%}"
        )


def print_hamiltonian_token_training_budget_summary(result: HamiltonianTokenTrainingBudgetResult) -> None:
    header = "steps    hidden   final loss   train mean   heldout mean   gen mean   token-gen   >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for row in summarize_hamiltonian_token_training_budget(result):
        print(
            f"{row.num_steps:<8} "
            f"{row.hidden:<8} "
            f"{row.final_loss:>10.6f}   "
            f"{row.train_mean_best:>10.4f}   "
            f"{row.heldout_mean_best:>12.4f}   "
            f"{row.generated_mean_best:>8.4f}   "
            f"{row.heldout_delta_vs_generated:>+9.4f}   "
            f"{row.heldout_success_95:>6.1%}   {row.heldout_success_98:>6.1%}   {row.heldout_success_99:>6.1%}"
        )


def print_hamiltonian_token_template_comparison(result: HamiltonianTokenTemplateComparisonResult) -> None:
    header = (
        "template                 CZs   slots   train   heldout   stacks   loss      "
        "heldout mean   gen mean   token-gen   >=0.95   >=0.98   >=0.99"
    )
    print(header)
    print("-" * len(header))
    for row in summarize_hamiltonian_token_template_comparison(result):
        print(
            f"{row.template:<24} "
            f"{row.n_entanglers:<5} "
            f"{row.n_slots:<7} "
            f"{row.n_train_targets:<7} "
            f"{row.n_heldout_targets:<8} "
            f"{row.n_solution_stacks:<7} "
            f"{row.final_loss:>8.5f}   "
            f"{row.heldout_mean_best:>12.4f}   "
            f"{row.generated_mean_best:>8.4f}   "
            f"{row.heldout_delta_vs_generated:>+9.4f}   "
            f"{row.heldout_success_95:>6.1%}   "
            f"{row.heldout_success_98:>6.1%}   "
            f"{row.heldout_success_99:>6.1%}"
        )


def _mean_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    std = tensor.std(unbiased=False) if tensor.numel() > 1 else torch.tensor(0.0)
    return float(tensor.mean().item()), float(std.item())


def print_hamiltonian_token_repeatability_summary(result: HamiltonianTokenRepeatabilityResult) -> None:
    rows = summarize_hamiltonian_token_repeatability(result)
    header = "run   steps   train   heldout   stacks   loss      train mean   heldout mean   gen mean   token-gen   >=0.95   >=0.98"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.run:<5} "
            f"{row.num_steps:<7} "
            f"{row.n_train_targets:<7} "
            f"{row.n_heldout_targets:<8} "
            f"{row.n_solution_stacks:<7} "
            f"{row.final_loss:>8.5f}   "
            f"{row.train_mean_best:>10.4f}   "
            f"{row.heldout_mean_best:>12.4f}   "
            f"{row.generated_mean_best:>8.4f}   "
            f"{row.heldout_delta_vs_generated:>+9.4f}   "
            f"{row.heldout_success_95:>6.1%}   {row.heldout_success_98:>6.1%}"
        )

    print()
    summary_header = "metric                    mean      std"
    print(summary_header)
    print("-" * len(summary_header))
    for label, values in [
        ("heldout mean", [row.heldout_mean_best for row in rows]),
        ("generated mean", [row.generated_mean_best for row in rows]),
        ("token-gen", [row.heldout_delta_vs_generated for row in rows]),
        (">=0.95", [row.heldout_success_95 for row in rows]),
        (">=0.98", [row.heldout_success_98 for row in rows]),
        ("final loss", [row.final_loss for row in rows]),
    ]:
        mean, std = _mean_std(values)
        print(f"{label:<24} {mean:>7.4f}   {std:>6.4f}")


def _repeatability_refinement_groups(
    result: HamiltonianRepeatabilityRefinementResult,
) -> dict[str, list[HamiltonianRepeatabilityRefinementRow]]:
    groups: dict[str, list[HamiltonianRepeatabilityRefinementRow]] = {}
    for row in result.rows:
        groups.setdefault(row.source, []).append(row)
    return groups


def print_hamiltonian_repeatability_refinement(
    result: HamiltonianRepeatabilityRefinementResult,
    max_rows: int | None = 10,
) -> None:
    header = "run   target      source             before   after    steps   move mean   move max"
    print(header)
    print("-" * len(header))
    rows = result.rows if max_rows is None else result.rows[:max_rows]
    for row in rows:
        steps = str(row.steps_to_threshold) if row.steps_to_threshold >= 0 else "miss"
        print(
            f"{row.run:<5} "
            f"{row.target:<11} "
            f"{row.source:<18} "
            f"{row.initial_fidelity:>6.4f}   "
            f"{row.refined_fidelity:>6.4f}   "
            f"{steps:>6}   "
            f"{row.movement_mean:>9.4f}   "
            f"{row.movement_max:>8.4f}"
        )
    if max_rows is not None and len(result.rows) > max_rows:
        print(f"... {len(result.rows) - max_rows} more")


def print_hamiltonian_repeatability_refinement_summary(
    result: HamiltonianRepeatabilityRefinementResult,
) -> None:
    header = "source             n   mean before   mean after   >=threshold   median steps   mean move   max move"
    print(header)
    print("-" * len(header))
    for source, rows in _repeatability_refinement_groups(result).items():
        before = torch.tensor([row.initial_fidelity for row in rows], dtype=torch.float32)
        after = torch.tensor([row.refined_fidelity for row in rows], dtype=torch.float32)
        mean_moves = torch.tensor([row.movement_mean for row in rows], dtype=torch.float32)
        max_moves = torch.tensor([row.movement_max for row in rows], dtype=torch.float32)
        reached = after >= result.threshold
        steps = torch.tensor(
            [row.steps_to_threshold for row in rows if row.steps_to_threshold >= 0],
            dtype=torch.float32,
        )
        median_steps = steps.median().item() if steps.numel() else float("nan")
        print(
            f"{source:<18} {len(rows):<3} "
            f"{before.mean().item():>11.4f}   "
            f"{after.mean().item():>10.4f}   "
            f"{reached.float().mean().item():>11.1%}   "
            f"{median_steps:>12.1f}   "
            f"{mean_moves.mean().item():>9.4f}   "
            f"{max_moves.max().item():>8.4f}"
        )


def print_hamiltonian_level1_headline_table(result: HamiltonianRepeatabilityRefinementResult) -> None:
    headline = summarize_hamiltonian_level1_headline(result)
    threshold_label = f">={headline.threshold:g}"
    header = (
        "source             n   proposal mean   run std   refined mean   "
        f"{threshold_label:<8}   median steps   mean move   max move"
    )
    print(header)
    print("-" * len(header))
    for row in headline.rows:
        print(
            f"{row.source:<18} {row.n_targets:<3} "
            f"{row.proposal_mean:>13.4f}   "
            f"{row.proposal_run_std:>7.4f}   "
            f"{row.refined_mean:>12.4f}   "
            f"{row.refinement_success:>8.1%}   "
            f"{row.median_steps:>12.1f}   "
            f"{row.mean_movement:>9.4f}   "
            f"{row.max_movement:>8.4f}"
        )
    print()
    print(
        "token - generated-search proposal advantage: "
        f"{headline.proposal_advantage_mean:+.4f} +/- {headline.proposal_advantage_std:.4f} "
        f"over {headline.n_runs} runs"
    )


def print_hamiltonian_denoise_diagnostic(result: HamiltonianDenoiseDiagnosticResult) -> None:
    header = "timestep   sigma    mse      zero mse  rel mse   cosine   target norm   pred norm"
    print(header)
    print("-" * len(header))
    for row in summarize_hamiltonian_denoise_diagnostic(result):
        print(
            f"{row.timestep:<9} "
            f"{row.sigma:>7.4f} "
            f"{row.mse:>8.4f} "
            f"{row.zero_mse:>9.4f} "
            f"{row.relative_mse:>8.4f} "
            f"{row.cosine:>8.4f} "
            f"{row.target_norm:>11.4f} "
            f"{row.pred_norm:>10.4f}"
        )


def print_hamiltonian_denoise_ablation(result: HamiltonianDenoiseAblationResult) -> None:
    header = (
        "config                         steps   hidden   final loss   "
        "rel mse t=1   rel mse t=T   cosine t=T   pred/target"
    )
    print(header)
    print("-" * len(header))
    for row in summarize_hamiltonian_denoise_ablation(result):
        print(
            f"{row.name:<30} "
            f"{row.num_steps:>5}   "
            f"{row.hidden:>6}   "
            f"{row.final_loss:>10.6f}   "
            f"{row.t1_relative_mse:>11.4f}   "
            f"{row.final_relative_mse:>11.4f}   "
            f"{row.final_cosine:>10.4f}   "
            f"{row.final_pred_target_norm_ratio:>11.4f}"
        )


def print_hamiltonian_denoise_normalization(result: HamiltonianDenoiseNormalizationResult) -> None:
    header = "variant             target scale   final loss   rel mse t=T   cosine t=T   pred/target"
    print(header)
    print("-" * len(header))
    for row in summarize_hamiltonian_denoise_normalization(result):
        print(
            f"{row.variant:<18} "
            f"{row.target_scale:>12.4f}   "
            f"{row.final_loss:>10.6f}   "
            f"{row.final_relative_mse:>11.4f}   "
            f"{row.final_cosine:>10.4f}   "
            f"{row.final_pred_target_norm_ratio:>11.4f}"
        )


def print_hamiltonian_skeleton_denoise_comparison(result: HamiltonianSkeletonDenoiseComparisonResult) -> None:
    header = "variant          final loss   rel mse t=T   cosine t=T   pred/target"
    print(header)
    print("-" * len(header))
    for row in summarize_hamiltonian_skeleton_denoise_comparison(result):
        print(
            f"{row.variant:<16} "
            f"{row.final_loss:>10.6f}   "
            f"{row.final_relative_mse:>11.4f}   "
            f"{row.final_cosine:>10.4f}   "
            f"{row.final_pred_target_norm_ratio:>11.4f}"
        )


def print_hamiltonian_slotwise_denoise_comparison(result: HamiltonianSlotwiseDenoiseComparisonResult) -> None:
    header = "variant          final loss   rel mse t=T   cosine t=T   pred/target"
    print(header)
    print("-" * len(header))
    for row in summarize_hamiltonian_slotwise_denoise_comparison(result):
        print(
            f"{row.variant:<16} "
            f"{row.final_loss:>10.6f}   "
            f"{row.final_relative_mse:>11.4f}   "
            f"{row.final_cosine:>10.4f}   "
            f"{row.final_pred_target_norm_ratio:>11.4f}"
        )


def print_hamiltonian_token_denoise_comparison(result: HamiltonianTokenDenoiseComparisonResult) -> None:
    header = "variant          final loss   rel mse t=T   cosine t=T   pred/target"
    print(header)
    print("-" * len(header))
    for row in summarize_hamiltonian_token_denoise_comparison(result):
        print(
            f"{row.variant:<16} "
            f"{row.final_loss:>10.6f}   "
            f"{row.final_relative_mse:>11.4f}   "
            f"{row.final_cosine:>10.4f}   "
            f"{row.final_pred_target_norm_ratio:>11.4f}"
        )


def print_hamiltonian_prior_search(result: HamiltonianPriorSearchResult, max_rows: int | None = 6) -> None:
    header = "target      uniform   prior    best prior labels"
    print(header)
    print("-" * len(header))
    rows = result.benchmarks if max_rows is None else result.benchmarks[:max_rows]
    for item in rows:
        labels = ", ".join(label if label is not None else "?" for label in item.prior_report.candidates[0].slot_labels)
        print(
            f"{item.target.name:<11} "
            f"{_best(item.uniform_report):>7.4f} "
            f"{_best(item.prior_report):>7.4f}   "
            f"{labels}"
        )
    if max_rows is not None and len(result.benchmarks) > max_rows:
        print(f"... {len(result.benchmarks) - max_rows} more")


def print_hamiltonian_prior_search_summary(result: HamiltonianPriorSearchResult) -> None:
    header = "mode                   n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_hamiltonian_prior_search(result):
        print(
            f"{item.mode:<22} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_hamiltonian_prior_mixture_summary(result: HamiltonianPriorMixtureResult) -> None:
    header = "alpha   n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in summarize_hamiltonian_prior_mixture(result):
        alpha = item.mode.removeprefix("alpha=")
        print(
            f"{alpha:<7} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_hamiltonian_mixture_refinement(result: HamiltonianMixtureRefinementResult, max_rows: int | None = 8) -> None:
    header = "target      alpha   before   after    steps"
    print(header)
    print("-" * len(header))
    rows = result.rows if max_rows is None else result.rows[:max_rows]
    for row in rows:
        steps = str(row.steps_to_threshold) if row.steps_to_threshold >= 0 else "miss"
        print(
            f"{row.target:<11} {row.alpha:<7g} "
            f"{row.initial_fidelity:>7.4f} "
            f"{row.refined_fidelity:>8.4f} "
            f"{steps:>8}"
        )
    if max_rows is not None and len(result.rows) > max_rows:
        print(f"... {len(result.rows) - max_rows} more")


def print_hamiltonian_mixture_refinement_summary(result: HamiltonianMixtureRefinementResult) -> None:
    header = "alpha   n   mean before   mean after   >=threshold   median steps"
    print(header)
    print("-" * len(header))
    for alpha, rows in _mixture_refinement_groups(result).items():
        initial = torch.tensor([row.initial_fidelity for row in rows], dtype=torch.float32)
        refined = torch.tensor([row.refined_fidelity for row in rows], dtype=torch.float32)
        steps = torch.tensor([row.steps_to_threshold for row in rows if row.steps_to_threshold >= 0], dtype=torch.float32)
        success = (refined >= result.threshold).float().mean().item()
        median_steps = steps.median().item() if steps.numel() else float("nan")
        print(
            f"{alpha:<7g} {len(rows):<3} "
            f"{initial.mean().item():>11.4f}   "
            f"{refined.mean().item():>10.4f}   "
            f"{success:>10.1%}   "
            f"{median_steps:>12.1f}"
        )


def print_hamiltonian_budget_refinement_summary(result: HamiltonianBudgetRefinementResult) -> None:
    header = "budget   alpha   n   mean before   mean after   >=threshold   min after"
    print(header)
    print("-" * len(header))
    for (budget, alpha), rows in _budget_refinement_groups(result).items():
        initial = torch.tensor([row.initial_fidelity for row in rows], dtype=torch.float32)
        refined = torch.tensor([row.refined_fidelity for row in rows], dtype=torch.float32)
        success = torch.tensor([row.reached_threshold for row in rows], dtype=torch.float32).mean().item()
        print(
            f"{budget:<8} {alpha:<7g} {len(rows):<3} "
            f"{initial.mean().item():>11.4f}   "
            f"{refined.mean().item():>10.4f}   "
            f"{success:>10.1%}   "
            f"{refined.min().item():>9.4f}"
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
    n_unique = len({target.name for target in dataset.targets})
    header = "n stacks   n targets   mean before   mean after   median gain   min after   >=0.99 after"
    print(header)
    print("-" * len(header))
    print(
        f"{len(dataset.targets):<3} "
        f"{n_unique:<9} "
        f"{dataset.initial_fidelities.mean().item():>11.4f}   "
        f"{dataset.refined_fidelities.mean().item():>10.4f}   "
        f"{gains.median().item():>11.4f}   "
        f"{dataset.refined_fidelities.min().item():>9.4f}   "
        f"{(dataset.refined_fidelities >= 0.99).float().mean().item():>10.1%}"
    )


def print_hamiltonian_template_comparison(result: HamiltonianTemplateComparisonResult) -> None:
    header = "template               CZs   slots   stacks   proposal   refined   >=threshold   median steps"
    print(header)
    print("-" * len(header))
    for row in result.rows:
        print(
            f"{row.template:<22} "
            f"{row.n_entanglers:<5} "
            f"{row.n_slots:<7} "
            f"{row.n_stacks:<6} "
            f"{row.proposal_mean:>8.4f}   "
            f"{row.refined_mean:>7.4f}   "
            f"{row.refinement_success:>10.1%}   "
            f"{row.median_steps:>12.1f}"
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


def _seed_ablation_groups(result: HamiltonianSeedAblationResult) -> dict[str, list[HamiltonianSeedAblationRow]]:
    groups: dict[str, list[HamiltonianSeedAblationRow]] = {}
    for row in result.rows:
        groups.setdefault(row.seed_type, []).append(row)
    return groups


def print_hamiltonian_seed_ablation(result: HamiltonianSeedAblationResult, max_rows: int | None = 8) -> None:
    header = "target      seed              before   after    steps"
    print(header)
    print("-" * len(header))
    rows = result.rows if max_rows is None else result.rows[:max_rows]
    for row in rows:
        steps = str(row.steps_to_threshold) if row.steps_to_threshold >= 0 else "miss"
        print(
            f"{row.target:<11} {row.seed_type:<17} "
            f"{row.initial_fidelity:>7.4f} "
            f"{row.refined_fidelity:>8.4f} "
            f"{steps:>8}"
        )
    if max_rows is not None and len(result.rows) > max_rows:
        print(f"... {len(result.rows) - max_rows} more")


def print_hamiltonian_seed_ablation_summary(result: HamiltonianSeedAblationResult) -> None:
    header = "seed              n   mean before   mean after   >=threshold   median steps"
    print(header)
    print("-" * len(header))
    for seed_type, rows in _seed_ablation_groups(result).items():
        initial = torch.tensor([row.initial_fidelity for row in rows], dtype=torch.float32)
        refined = torch.tensor([row.refined_fidelity for row in rows], dtype=torch.float32)
        steps = torch.tensor([row.steps_to_threshold for row in rows if row.steps_to_threshold >= 0], dtype=torch.float32)
        success = (refined >= result.threshold).float().mean().item()
        median_steps = steps.median().item() if steps.numel() else float("nan")
        print(
            f"{seed_type:<17} {len(rows):<3} "
            f"{initial.mean().item():>11.4f}   "
            f"{refined.mean().item():>10.4f}   "
            f"{success:>10.1%}   "
            f"{median_steps:>12.1f}"
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


def plot_hamiltonian_seed_ablation(result: HamiltonianSeedAblationResult) -> None:
    groups = _seed_ablation_groups(result)
    labels = list(groups)
    before = [[row.initial_fidelity for row in groups[label]] for label in labels]
    after = [[row.refined_fidelity for row in groups[label]] for label in labels]
    step_values = []
    for label in labels:
        reached = [row.steps_to_threshold for row in groups[label] if row.steps_to_threshold >= 0]
        step_values.append(reached if reached else [float("nan")])

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    positions_before = [i * 3 + 1 for i in range(len(labels))]
    positions_after = [i * 3 + 2 for i in range(len(labels))]
    plt.boxplot(before, positions=positions_before, widths=0.7, showmeans=True)
    plt.boxplot(after, positions=positions_after, widths=0.7, showmeans=True)
    plt.xticks([(a + b) / 2 for a, b in zip(positions_before, positions_after)], labels, rotation=20)
    plt.ylabel("unitary fidelity")
    plt.title("Before vs after refinement")
    plt.ylim(0.0, 1.02)

    plt.subplot(1, 2, 2)
    plt.boxplot(step_values, labels=labels, showmeans=True)
    plt.ylabel(f"steps to F >= {result.threshold:g}")
    plt.title("Refinement speed")
    plt.xticks(rotation=20)
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


def plot_hamiltonian_conditioned_diffusion(
    baseline: HamiltonianSuiteResult,
    conditioned: HamiltonianConditionedDiffusionResult,
) -> None:
    if not baseline.benchmarks:
        raise ValueError("baseline must contain at least one benchmark")
    if len(baseline.benchmarks) != len(conditioned.reports):
        raise ValueError("baseline and conditioned result must cover the same number of targets")
    values = [
        [_best(item.clifford_report) for item in baseline.benchmarks],
        [_best(item.analytic_report) for item in baseline.benchmarks],
        [_best(item.generated_report) for item in baseline.benchmarks],
        [_best(item.haar_report) for item in baseline.benchmarks],
        [_best(report) for report in conditioned.reports],
    ]
    labels = ["Clifford", "analytic", "generated", "Haar", "H-cond diffusion"]
    plt.figure(figsize=(10, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Hamiltonian-conditioned SU(2)^6 diffusion proposals")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hamiltonian_token_conditioned_diffusion(
    baseline: HamiltonianSuiteResult,
    conditioned: HamiltonianConditionedDiffusionResult,
) -> None:
    if not baseline.benchmarks:
        raise ValueError("baseline must contain at least one benchmark")
    if len(baseline.benchmarks) != len(conditioned.reports):
        raise ValueError("baseline and conditioned result must cover the same number of targets")
    values = [
        [_best(item.clifford_report) for item in baseline.benchmarks],
        [_best(item.analytic_report) for item in baseline.benchmarks],
        [_best(item.generated_report) for item in baseline.benchmarks],
        [_best(item.haar_report) for item in baseline.benchmarks],
        [_best(report) for report in conditioned.reports],
    ]
    labels = ["Clifford", "analytic", "generated", "Haar", "token diffusion"]
    plt.figure(figsize=(10, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Hamiltonian circuit-token SU(2)^6 diffusion proposals")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hamiltonian_token_heldout_comparison(
    heldout_baseline: HamiltonianSuiteResult,
    token_diagnostic: HamiltonianConditionedOverfitDiagnosticResult,
) -> None:
    if not heldout_baseline.benchmarks:
        raise ValueError("heldout_baseline must contain at least one benchmark")
    if len(heldout_baseline.benchmarks) != len(token_diagnostic.heldout_reports):
        raise ValueError("heldout baseline and token diagnostic must cover the same number of targets")
    values = [
        [_best(item.clifford_report) for item in heldout_baseline.benchmarks],
        [_best(item.analytic_report) for item in heldout_baseline.benchmarks],
        [_best(item.generated_report) for item in heldout_baseline.benchmarks],
        [_best(item.haar_report) for item in heldout_baseline.benchmarks],
        [_best(report) for report in token_diagnostic.heldout_reports],
    ]
    labels = ["Clifford", "analytic", "generated", "Haar", "token heldout"]
    plt.figure(figsize=(10, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Held-out Hamiltonian circuit-token proposals")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hamiltonian_token_data_scale(result: HamiltonianTokenDataScaleResult) -> None:
    rows = summarize_hamiltonian_token_data_scale(result)
    if not rows:
        raise ValueError("result must contain at least one data-scale row")
    counts = [row.n_train_targets for row in rows]
    train = [row.train_mean_best for row in rows]
    heldout = [row.heldout_mean_best for row in rows]
    generated = [row.generated_mean_best for row in rows]
    success_95 = [row.heldout_success_95 for row in rows]
    success_98 = [row.heldout_success_98 for row in rows]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    ax0.plot(counts, train, marker="o", label="train targets")
    ax0.plot(counts, heldout, marker="o", label="heldout targets")
    ax0.plot(counts, generated, linestyle="--", color="tab:gray", label="generated-search baseline")
    ax0.set_xlabel("number of training Hamiltonians")
    ax0.set_ylabel("mean best unitary fidelity")
    ax0.set_ylim(0.0, 1.02)
    ax0.legend()

    ax1.plot(counts, success_95, marker="o", label="heldout >= 0.95")
    ax1.plot(counts, success_98, marker="o", label="heldout >= 0.98")
    ax1.set_xlabel("number of training Hamiltonians")
    ax1.set_ylabel("heldout success fraction")
    ax1.set_ylim(0.0, 1.02)
    ax1.legend()

    if len(counts) > 1:
        ax0.set_xscale("log", base=2)
        ax1.set_xscale("log", base=2)
    fig.suptitle("Hamiltonian circuit-token data scale-up")
    fig.tight_layout()


def plot_hamiltonian_token_training_budget(result: HamiltonianTokenTrainingBudgetResult) -> None:
    rows = summarize_hamiltonian_token_training_budget(result)
    if not rows:
        raise ValueError("result must contain at least one training-budget row")
    steps = [row.num_steps for row in rows]
    train = [row.train_mean_best for row in rows]
    heldout = [row.heldout_mean_best for row in rows]
    generated = [row.generated_mean_best for row in rows]
    losses = [row.final_loss for row in rows]
    success_95 = [row.heldout_success_95 for row in rows]
    success_98 = [row.heldout_success_98 for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps, losses, marker="o")
    axes[0].set_xlabel("training steps")
    axes[0].set_ylabel("final denoising loss")
    axes[0].set_title("Optimization")

    axes[1].plot(steps, train, marker="o", label="train targets")
    axes[1].plot(steps, heldout, marker="o", label="heldout targets")
    axes[1].plot(steps, generated, linestyle="--", color="tab:gray", label="generated-search baseline")
    axes[1].set_xlabel("training steps")
    axes[1].set_ylabel("mean best unitary fidelity")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Proposal quality")
    axes[1].legend()

    axes[2].plot(steps, success_95, marker="o", label="heldout >= 0.95")
    axes[2].plot(steps, success_98, marker="o", label="heldout >= 0.98")
    axes[2].set_xlabel("training steps")
    axes[2].set_ylabel("heldout success fraction")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("Held-out success")
    axes[2].legend()

    if len(steps) > 1:
        for ax in axes:
            ax.set_xscale("log", base=2)
    fig.suptitle("Hamiltonian circuit-token training budget")
    fig.tight_layout()


def _latest_token_diagnostic(
    result: HamiltonianTokenTrainingBudgetResult,
) -> HamiltonianConditionedOverfitDiagnosticResult:
    if not result.diagnostics:
        raise ValueError("training-budget result must contain at least one diagnostic")
    return result.diagnostics[max(result.diagnostics)]


def plot_hamiltonian_token_template_comparison(result: HamiltonianTokenTemplateComparisonResult) -> None:
    two_diagnostic = _latest_token_diagnostic(result.two_entangler)
    three_diagnostic = _latest_token_diagnostic(result.three_entangler)
    values = [
        [_best(item.generated_report) for item in result.two_entangler.heldout_baseline.benchmarks],
        [_best(report) for report in two_diagnostic.heldout_reports],
        [_best(item.generated_report) for item in result.three_entangler.heldout_baseline.benchmarks],
        [_best(report) for report in three_diagnostic.heldout_reports],
    ]
    labels = ["2 CZ generated", "2 CZ token", "3 CZ generated", "3 CZ token"]

    plt.figure(figsize=(8, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Hamiltonian token diffusion: 2 CZ vs 3 CZ")
    plt.ylim(0.0, 1.02)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()


def plot_hamiltonian_token_repeatability(result: HamiltonianTokenRepeatabilityResult) -> None:
    rows = summarize_hamiltonian_token_repeatability(result)
    if not rows:
        raise ValueError("result must contain at least one repeatability row")
    runs = [row.run for row in rows]
    heldout = [row.heldout_mean_best for row in rows]
    generated = [row.generated_mean_best for row in rows]
    delta = [row.heldout_delta_vs_generated for row in rows]
    success_95 = [row.heldout_success_95 for row in rows]
    success_98 = [row.heldout_success_98 for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(runs, heldout, marker="o", label="token heldout")
    axes[0].plot(runs, generated, marker="o", label="generated-search baseline")
    axes[0].set_xlabel("repeatability run")
    axes[0].set_ylabel("mean best unitary fidelity")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Held-out proposal quality")
    axes[0].legend()

    axes[1].axhline(0.0, linestyle="--", color="tab:gray")
    axes[1].bar(runs, delta)
    axes[1].set_xlabel("repeatability run")
    axes[1].set_ylabel("token - generated baseline")
    axes[1].set_title("Baseline advantage")

    axes[2].plot(runs, success_95, marker="o", label="heldout >= 0.95")
    axes[2].plot(runs, success_98, marker="o", label="heldout >= 0.98")
    axes[2].set_xlabel("repeatability run")
    axes[2].set_ylabel("success fraction")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("Held-out success")
    axes[2].legend()

    fig.suptitle("Hamiltonian circuit-token repeatability")
    fig.tight_layout()


def plot_hamiltonian_repeatability_refinement(
    result: HamiltonianRepeatabilityRefinementResult,
) -> None:
    groups = _repeatability_refinement_groups(result)
    if not groups:
        raise ValueError("result must contain at least one refinement row")
    labels = list(groups)
    before = [[row.initial_fidelity for row in groups[label]] for label in labels]
    after = [[row.refined_fidelity for row in groups[label]] for label in labels]
    movement = [[row.movement_mean for row in groups[label]] for label in labels]
    steps = []
    for label in labels:
        reached = [row.steps_to_threshold for row in groups[label] if row.steps_to_threshold >= 0]
        steps.append(reached if reached else [float("nan")])

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4))
    positions_before = [i + 1 - 0.15 for i in range(len(labels))]
    positions_after = [i + 1 + 0.15 for i in range(len(labels))]
    ax0.boxplot(before, positions=positions_before, widths=0.25, showmeans=True)
    ax0.boxplot(after, positions=positions_after, widths=0.25, showmeans=True)
    ax0.set_xticks(range(1, len(labels) + 1), labels)
    ax0.set_ylabel("unitary fidelity")
    ax0.set_ylim(0.0, 1.02)
    ax0.set_title("Before vs after refinement")
    ax0.legend(
        [
            plt.Line2D([0], [0], color="tab:blue"),
            plt.Line2D([0], [0], color="tab:orange"),
        ],
        ["before", "after"],
        loc="lower right",
    )

    ax1.boxplot(steps, labels=labels, showmeans=True)
    ax1.set_ylabel(f"steps to F >= {result.threshold:g}")
    ax1.set_title("Refinement speed")

    ax2.boxplot(movement, labels=labels, showmeans=True)
    ax2.set_ylabel("mean SU(2) movement")
    ax2.set_title("Proposal movement")
    fig.suptitle("Token vs generated-search refinement basins")
    fig.tight_layout()


def plot_hamiltonian_conditioned_overfit_diagnostic(
    result: HamiltonianConditionedOverfitDiagnosticResult,
) -> None:
    values = [
        [_best(report) for report in result.train_reports],
        [_best(report) for report in result.heldout_reports],
    ]
    plt.figure(figsize=(7, 4))
    plt.boxplot(values, labels=["train targets", "heldout targets"], showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Hamiltonian-conditioned diffusion overfit diagnostic")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hamiltonian_denoise_diagnostic(result: HamiltonianDenoiseDiagnosticResult) -> None:
    rows = summarize_hamiltonian_denoise_diagnostic(result)
    timesteps = [row.timestep for row in rows]
    rel_mse = [row.relative_mse for row in rows]
    cosine = [row.cosine for row in rows]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(timesteps, rel_mse, marker="o")
    plt.xlabel("diffusion timestep")
    plt.ylabel("MSE / zero-predictor MSE")
    plt.title("Denoising target fit")
    plt.ylim(bottom=0.0)

    plt.subplot(1, 2, 2)
    plt.plot(timesteps, cosine, marker="o")
    plt.xlabel("diffusion timestep")
    plt.ylabel("mean cosine")
    plt.title("Predicted tangent alignment")
    plt.ylim(-1.0, 1.0)
    plt.tight_layout()


def plot_hamiltonian_denoise_ablation(result: HamiltonianDenoiseAblationResult) -> None:
    rows = summarize_hamiltonian_denoise_ablation(result)
    labels = [row.name.replace(result.diagnostics[0].config.name.rsplit("-", 1)[0] + "-", "") for row in rows]
    rel_mse = [row.final_relative_mse for row in rows]
    cosine = [row.final_cosine for row in rows]
    ratio = [row.final_pred_target_norm_ratio for row in rows]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.bar(labels, rel_mse)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("MSE / zero-predictor MSE")
    plt.title("Final-step denoising")
    plt.xticks(rotation=20, ha="right")

    plt.subplot(1, 3, 2)
    plt.bar(labels, cosine)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("mean cosine")
    plt.title("Final-step alignment")
    plt.ylim(-1.0, 1.0)
    plt.xticks(rotation=20, ha="right")

    plt.subplot(1, 3, 3)
    plt.bar(labels, ratio)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("pred norm / target norm")
    plt.title("Output scale")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()


def plot_hamiltonian_denoise_normalization(result: HamiltonianDenoiseNormalizationResult) -> None:
    rows = summarize_hamiltonian_denoise_normalization(result)
    labels = [row.variant for row in rows]
    rel_mse = [row.final_relative_mse for row in rows]
    cosine = [row.final_cosine for row in rows]
    ratio = [row.final_pred_target_norm_ratio for row in rows]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.bar(labels, rel_mse)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("MSE / zero-predictor MSE")
    plt.title("Normalized target fit")
    plt.xticks(rotation=20, ha="right")

    plt.subplot(1, 3, 2)
    plt.bar(labels, cosine)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("mean cosine")
    plt.title("Tangent alignment")
    plt.ylim(-1.0, 1.0)
    plt.xticks(rotation=20, ha="right")

    plt.subplot(1, 3, 3)
    plt.bar(labels, ratio)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("pred norm / target norm")
    plt.title("Output scale")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()


def plot_hamiltonian_skeleton_denoise_comparison(result: HamiltonianSkeletonDenoiseComparisonResult) -> None:
    rows = summarize_hamiltonian_skeleton_denoise_comparison(result)
    labels = [row.variant for row in rows]
    rel_mse = [row.final_relative_mse for row in rows]
    cosine = [row.final_cosine for row in rows]
    ratio = [row.final_pred_target_norm_ratio for row in rows]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.bar(labels, rel_mse)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("MSE / zero-predictor MSE")
    plt.title("Skeleton denoising")
    plt.xticks(rotation=15, ha="right")

    plt.subplot(1, 3, 2)
    plt.bar(labels, cosine)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("mean cosine")
    plt.title("Tangent alignment")
    plt.ylim(-1.0, 1.0)
    plt.xticks(rotation=15, ha="right")

    plt.subplot(1, 3, 3)
    plt.bar(labels, ratio)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("pred norm / target norm")
    plt.title("Output scale")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()


def plot_hamiltonian_slotwise_denoise_comparison(result: HamiltonianSlotwiseDenoiseComparisonResult) -> None:
    rows = summarize_hamiltonian_slotwise_denoise_comparison(result)
    labels = [row.variant for row in rows]
    rel_mse = [row.final_relative_mse for row in rows]
    cosine = [row.final_cosine for row in rows]
    ratio = [row.final_pred_target_norm_ratio for row in rows]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.bar(labels, rel_mse)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("MSE / zero-predictor MSE")
    plt.title("Slot-wise denoising")
    plt.xticks(rotation=15, ha="right")

    plt.subplot(1, 3, 2)
    plt.bar(labels, cosine)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("mean cosine")
    plt.title("Tangent alignment")
    plt.ylim(-1.0, 1.0)
    plt.xticks(rotation=15, ha="right")

    plt.subplot(1, 3, 3)
    plt.bar(labels, ratio)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("pred norm / target norm")
    plt.title("Output scale")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()


def plot_hamiltonian_token_denoise_comparison(result: HamiltonianTokenDenoiseComparisonResult) -> None:
    rows = summarize_hamiltonian_token_denoise_comparison(result)
    labels = [row.variant for row in rows]
    rel_mse = [row.final_relative_mse for row in rows]
    cosine = [row.final_cosine for row in rows]
    ratio = [row.final_pred_target_norm_ratio for row in rows]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.bar(labels, rel_mse)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("MSE / zero-predictor MSE")
    plt.title("Circuit-token denoising")
    plt.xticks(rotation=15, ha="right")

    plt.subplot(1, 3, 2)
    plt.bar(labels, cosine)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("mean cosine")
    plt.title("Tangent alignment")
    plt.ylim(-1.0, 1.0)
    plt.xticks(rotation=15, ha="right")

    plt.subplot(1, 3, 3)
    plt.bar(labels, ratio)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("pred norm / target norm")
    plt.title("Output scale")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()


def plot_hamiltonian_prior_search(result: HamiltonianPriorSearchResult) -> None:
    if not result.benchmarks:
        raise ValueError("result must contain at least one benchmark")
    values = [
        [_best(item.uniform_report) for item in result.benchmarks],
        [_best(item.prior_report) for item in result.benchmarks],
    ]
    plt.figure(figsize=(7, 4))
    plt.boxplot(values, labels=["uniform generated", "learned prior"], showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Hamiltonian learned-prior search")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hamiltonian_prior_mixture(result: HamiltonianPriorMixtureResult) -> None:
    if not result.alpha_results:
        raise ValueError("result must contain at least one alpha")
    alphas = list(result.alpha_results)
    values = [
        [_best(item.prior_report) for item in result.alpha_results[alpha].benchmarks]
        for alpha in alphas
    ]
    plt.figure(figsize=(8, 4))
    plt.boxplot(values, labels=[f"{alpha:g}" for alpha in alphas], showmeans=True)
    plt.xlabel("prior mixture alpha")
    plt.ylabel("best unitary fidelity")
    plt.title("Hamiltonian prior-mixture search")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hamiltonian_mixture_refinement(result: HamiltonianMixtureRefinementResult) -> None:
    groups = _mixture_refinement_groups(result)
    alphas = list(groups)
    before = [[row.initial_fidelity for row in groups[alpha]] for alpha in alphas]
    after = [[row.refined_fidelity for row in groups[alpha]] for alpha in alphas]
    step_values = []
    for alpha in alphas:
        reached = [row.steps_to_threshold for row in groups[alpha] if row.steps_to_threshold >= 0]
        step_values.append(reached if reached else [float("nan")])

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    positions_before = [i * 3 + 1 for i in range(len(alphas))]
    positions_after = [i * 3 + 2 for i in range(len(alphas))]
    plt.boxplot(before, positions=positions_before, widths=0.7, showmeans=True)
    plt.boxplot(after, positions=positions_after, widths=0.7, showmeans=True)
    plt.xticks(
        [(a + b) / 2 for a, b in zip(positions_before, positions_after)],
        [f"{alpha:g}" for alpha in alphas],
    )
    plt.xlabel("prior mixture alpha")
    plt.ylabel("unitary fidelity")
    plt.title("Before vs after refinement")
    plt.ylim(0.0, 1.02)

    plt.subplot(1, 2, 2)
    plt.boxplot(step_values, labels=[f"{alpha:g}" for alpha in alphas], showmeans=True)
    plt.xlabel("prior mixture alpha")
    plt.ylabel(f"steps to F >= {result.threshold:g}")
    plt.title("Refinement speed")
    plt.tight_layout()


def plot_hamiltonian_budget_refinement(result: HamiltonianBudgetRefinementResult) -> None:
    if not result.rows:
        raise ValueError("result must contain at least one row")
    groups = _budget_refinement_groups(result)
    budgets = sorted({budget for budget, _ in groups})
    alphas = sorted({alpha for _, alpha in groups})

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for alpha in alphas:
        means = []
        for budget in budgets:
            rows = groups[(budget, alpha)]
            means.append(torch.tensor([row.refined_fidelity for row in rows], dtype=torch.float32).mean().item())
        plt.plot(budgets, means, marker="o", label=f"alpha={alpha:g}")
    plt.xlabel("refinement steps")
    plt.ylabel("mean refined fidelity")
    plt.title("Budgeted refinement quality")
    plt.ylim(0.0, 1.02)
    plt.legend()

    plt.subplot(1, 2, 2)
    for alpha in alphas:
        rates = []
        for budget in budgets:
            rows = groups[(budget, alpha)]
            rates.append(torch.tensor([row.reached_threshold for row in rows], dtype=torch.float32).mean().item())
        plt.plot(budgets, rates, marker="o", label=f"alpha={alpha:g}")
    plt.xlabel("refinement steps")
    plt.ylabel(f"fraction with F >= {result.threshold:g}")
    plt.title("Budgeted success rate")
    plt.ylim(0.0, 1.02)
    plt.legend()
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


def plot_hamiltonian_template_comparison(result: HamiltonianTemplateComparisonResult) -> None:
    values = [
        result.two_entangler.initial_fidelities.detach().cpu().tolist(),
        result.three_entangler.initial_fidelities.detach().cpu().tolist(),
        result.two_entangler.refined_fidelities.detach().cpu().tolist(),
        result.three_entangler.refined_fidelities.detach().cpu().tolist(),
    ]
    labels = ["2 CZ proposal", "3 CZ proposal", "2 CZ refined", "3 CZ refined"]
    plt.figure(figsize=(8, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("unitary fidelity")
    plt.title("Two-qubit template comparison")
    plt.ylim(0.0, 1.02)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
