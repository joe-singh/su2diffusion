from dataclasses import dataclass
import re

import matplotlib.pyplot as plt
import torch

from .quaternion import sample_haar
from .synthesis import (
    HiddenShallowCircuitAggregate,
    SynthesisReport,
    sample_near_clifford_gates,
    synthesize_unitary_two_entangler_random_report,
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
