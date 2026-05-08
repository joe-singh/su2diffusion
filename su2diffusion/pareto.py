from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import torch

from .hamiltonian import (
    HamiltonianTarget,
    ThreeQubitCZTemplate,
    get_three_qubit_cz_template,
    refine_three_qubit_template_candidate,
    synthesize_three_qubit_template_random_report,
)
from .quaternion import q_normalize, su2_distance
from .synthesis import RefinementResult, SynthesisCandidate, SynthesisReport


@dataclass(frozen=True)
class ParetoScoringConfig:
    """Post-hoc score weights for hardware-cost-aware circuit ranking."""

    cz_weight: float = 0.015
    local_gate_weight: float = 0.0005
    movement_weight: float = 0.04
    angle_weight: float = 0.001


@dataclass(frozen=True)
class ParetoCandidateRow:
    target: str
    template: str
    source: str
    candidate_rank: int
    n_cz: int
    n_local_gates: int
    proposal_fidelity: float
    refined_fidelity: float
    steps_to_threshold: int
    movement_mean: float
    movement_max: float
    local_angle_sum: float
    hardware_cost: float
    regularized_score: float
    is_pareto: bool
    slot_indices: tuple[int, ...]
    slot_labels: tuple[str | None, ...]
    refined_gates: torch.Tensor


@dataclass(frozen=True)
class ParetoCircuitResult:
    target: HamiltonianTarget
    templates: tuple[ThreeQubitCZTemplate, ...]
    source: str
    scoring: ParetoScoringConfig
    rows: list[ParetoCandidateRow]
    reports: dict[str, SynthesisReport]
    threshold: float


def _coerce_template(template: str | ThreeQubitCZTemplate) -> ThreeQubitCZTemplate:
    if isinstance(template, ThreeQubitCZTemplate):
        return template
    return get_three_qubit_cz_template(template)


def _steps_to_threshold(initial: float, trace: tuple[float, ...], threshold: float) -> int:
    if initial >= threshold:
        return 0
    for step, value in enumerate(trace, start=1):
        if value >= threshold:
            return step
    return -1


def _local_angle_sum(q_stack: torch.Tensor) -> float:
    q_stack = q_normalize(q_stack)
    w = q_stack[..., 0].abs().clamp(max=1.0)
    angles = 2.0 * torch.acos(w)
    return float(angles.sum().detach().cpu())


def pareto_hardware_cost(
    n_cz: int,
    n_local_gates: int,
    movement_mean: float,
    local_angle_sum: float,
    scoring: ParetoScoringConfig | None = None,
) -> float:
    scoring = scoring or ParetoScoringConfig()
    return (
        scoring.cz_weight * n_cz
        + scoring.local_gate_weight * n_local_gates
        + scoring.movement_weight * movement_mean
        + scoring.angle_weight * local_angle_sum
    )


def _refinement_movement(
    start_stack: torch.Tensor,
    refined_stack: torch.Tensor,
) -> tuple[tuple[float, ...], float, float]:
    distances = su2_distance(q_normalize(start_stack), q_normalize(refined_stack)).detach().cpu()
    return (
        tuple(float(item) for item in distances.tolist()),
        float(distances.mean().item()),
        float(distances.max().item()),
    )


def _candidate_start_stack(local_gates: torch.Tensor, candidate: SynthesisCandidate) -> torch.Tensor:
    return q_normalize(local_gates[list(candidate.slot_indices)]).detach()


def _make_pareto_row(
    target: HamiltonianTarget,
    template: ThreeQubitCZTemplate,
    source: str,
    rank: int,
    candidate: SynthesisCandidate,
    start_stack: torch.Tensor,
    refinement: RefinementResult,
    scoring: ParetoScoringConfig,
    threshold: float,
) -> ParetoCandidateRow:
    _, movement_mean, movement_max = _refinement_movement(start_stack, refinement.refined_gates)
    local_angle_sum = _local_angle_sum(refinement.refined_gates)
    hardware_cost = pareto_hardware_cost(
        n_cz=len(template.edges),
        n_local_gates=template.n_slots,
        movement_mean=movement_mean,
        local_angle_sum=local_angle_sum,
        scoring=scoring,
    )
    return ParetoCandidateRow(
        target=target.name,
        template=template.name,
        source=source,
        candidate_rank=rank,
        n_cz=len(template.edges),
        n_local_gates=template.n_slots,
        proposal_fidelity=float(candidate.fidelity),
        refined_fidelity=float(refinement.refined_fidelity),
        steps_to_threshold=_steps_to_threshold(
            refinement.initial_fidelity,
            refinement.fidelity_trace,
            threshold,
        ),
        movement_mean=movement_mean,
        movement_max=movement_max,
        local_angle_sum=local_angle_sum,
        hardware_cost=hardware_cost,
        regularized_score=float(refinement.refined_fidelity - hardware_cost),
        is_pareto=False,
        slot_indices=candidate.slot_indices,
        slot_labels=candidate.slot_labels,
        refined_gates=refinement.refined_gates.detach(),
    )


def _mark_pareto_frontier(rows: list[ParetoCandidateRow]) -> list[ParetoCandidateRow]:
    marked = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                other.hardware_cost <= row.hardware_cost + 1e-12
                and other.refined_fidelity >= row.refined_fidelity - 1e-12
            )
            strictly_better = (
                other.hardware_cost < row.hardware_cost - 1e-12
                or other.refined_fidelity > row.refined_fidelity + 1e-12
            )
            if no_worse and strictly_better:
                dominated = True
                break
        marked.append(replace(row, is_pareto=not dominated))
    return marked


def rescore_pareto_circuit_result(
    result: ParetoCircuitResult,
    scoring: ParetoScoringConfig,
) -> ParetoCircuitResult:
    """Recompute cost/score/frontier without rerunning search or refinement."""

    rows = []
    for row in result.rows:
        hardware_cost = pareto_hardware_cost(
            n_cz=row.n_cz,
            n_local_gates=row.n_local_gates,
            movement_mean=row.movement_mean,
            local_angle_sum=row.local_angle_sum,
            scoring=scoring,
        )
        rows.append(
            replace(
                row,
                hardware_cost=hardware_cost,
                regularized_score=float(row.refined_fidelity - hardware_cost),
                is_pareto=False,
            )
        )
    return replace(result, scoring=scoring, rows=_mark_pareto_frontier(rows))


def run_pareto_circuit_sampling(
    target: HamiltonianTarget,
    generated_gates: torch.Tensor,
    generated_labels: list[str | None] | None = None,
    templates: tuple[str | ThreeQubitCZTemplate, ...] | list[str | ThreeQubitCZTemplate] = (
        "line-3cz-a",
        "line-3cz-b",
        "line-4cz",
    ),
    source: str = "generated-search",
    n_random_candidates: int = 2_000,
    top_k_per_template: int = 6,
    refinement_steps: int = 60,
    refinement_lr: float = 0.05,
    threshold: float = 0.99,
    scoring: ParetoScoringConfig | None = None,
    seed: int = 0,
    show_progress: bool = True,
) -> ParetoCircuitResult:
    """Search/refine candidate circuits and score the fidelity/cost tradeoff.

    This is deliberately post-hoc: it does not change the diffusion loss. It asks
    which candidates from the generated local-gate pool are attractive after
    accounting for template depth, local-gate count, and refinement movement.
    """

    if generated_gates.shape[0] == 0:
        raise ValueError("generated_gates must contain at least one gate")
    if generated_labels is not None and len(generated_labels) != generated_gates.shape[0]:
        raise ValueError("generated_labels must have one entry per generated gate")
    if n_random_candidates <= 0:
        raise ValueError("n_random_candidates must be positive")
    if top_k_per_template <= 0:
        raise ValueError("top_k_per_template must be positive")
    if refinement_steps <= 0:
        raise ValueError("refinement_steps must be positive")
    if refinement_lr <= 0:
        raise ValueError("refinement_lr must be positive")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    scoring = scoring or ParetoScoringConfig()
    template_objs = tuple(_coerce_template(template) for template in templates)
    generated_gates = q_normalize(generated_gates)
    reports: dict[str, SynthesisReport] = {}
    rows: list[ParetoCandidateRow] = []

    iterator = list(enumerate(template_objs))
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc=f"Pareto sampling {target.name}", dynamic_ncols=True)

    for template_index, template in iterator:
        if target.unitary.shape != (2**template.n_qubits, 2**template.n_qubits):
            raise ValueError(f"target must be a {template.n_qubits}-qubit Hamiltonian target")
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(template=template.name)
        report = synthesize_three_qubit_template_random_report(
            generated_gates,
            target_unitary=target.unitary,
            target_name=target.name,
            template=template,
            n_candidates=n_random_candidates,
            top_k=top_k_per_template,
            local_labels=generated_labels,
            seed=seed + 10_000 * template_index,
            name=f"{target.name} {template.name} Pareto candidates",
            mode=source,
            keep_fidelities=False,
        )
        reports[template.name] = report
        for rank, candidate in enumerate(report.candidates, start=1):
            start_stack = _candidate_start_stack(generated_gates, candidate)
            refinement = refine_three_qubit_template_candidate(
                generated_gates,
                candidate,
                target_unitary=target.unitary,
                template=template,
                num_steps=refinement_steps,
                lr=refinement_lr,
            )
            rows.append(
                _make_pareto_row(
                    target=target,
                    template=template,
                    source=source,
                    rank=rank,
                    candidate=candidate,
                    start_stack=start_stack,
                    refinement=refinement,
                    scoring=scoring,
                    threshold=threshold,
                )
            )

    rows = _mark_pareto_frontier(rows)
    return ParetoCircuitResult(
        target=target,
        templates=template_objs,
        source=source,
        scoring=scoring,
        rows=rows,
        reports=reports,
        threshold=threshold,
    )


def pareto_frontier_rows(result: ParetoCircuitResult) -> list[ParetoCandidateRow]:
    return sorted(
        [row for row in result.rows if row.is_pareto],
        key=lambda row: (row.hardware_cost, -row.refined_fidelity),
    )


def top_pareto_rows(
    result: ParetoCircuitResult,
    max_rows: int | None = 10,
) -> list[ParetoCandidateRow]:
    rows = sorted(result.rows, key=lambda row: (-row.regularized_score, -row.refined_fidelity))
    return rows if max_rows is None else rows[:max_rows]


def print_pareto_circuit_candidates(
    result: ParetoCircuitResult,
    max_rows: int | None = 12,
) -> None:
    header = "frontier template   CZs   proposal   refined   cost    score    steps"
    print(header)
    print("-" * len(header))
    for row in top_pareto_rows(result, max_rows=max_rows):
        steps = str(row.steps_to_threshold) if row.steps_to_threshold >= 0 else "miss"
        marker = "*" if row.is_pareto else ""
        print(
            f"{marker:<8} "
            f"{row.template:<10} "
            f"{row.n_cz:<5} "
            f"{row.proposal_fidelity:>8.4f}   "
            f"{row.refined_fidelity:>7.4f}   "
            f"{row.hardware_cost:>5.3f}   "
            f"{row.regularized_score:>6.4f}   "
            f"{steps:>5}"
        )
    if max_rows is not None and len(result.rows) > max_rows:
        print(f"... {len(result.rows) - max_rows} more")


def print_pareto_circuit_summary(result: ParetoCircuitResult) -> None:
    header = "template   n   CZs   best F   mean F   best score   frontier"
    print(header)
    print("-" * len(header))
    for template in [item.name for item in result.templates]:
        group = [row for row in result.rows if row.template == template]
        if not group:
            continue
        refined = torch.tensor([row.refined_fidelity for row in group], dtype=torch.float32)
        best_score = max(row.regularized_score for row in group)
        frontier = sum(row.is_pareto for row in group)
        print(
            f"{template:<10} "
            f"{len(group):<3} "
            f"{group[0].n_cz:<5} "
            f"{refined.max().item():>6.4f}   "
            f"{refined.mean().item():>6.4f}   "
            f"{best_score:>10.4f}   "
            f"{frontier:<8}"
        )


def plot_pareto_circuit_sampling(result: ParetoCircuitResult) -> None:
    if not result.rows:
        raise ValueError("result must contain at least one candidate row")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    templates = [template.name for template in result.templates]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_by_template = {template: colors[i % len(colors)] for i, template in enumerate(templates)}

    for template in templates:
        group = [row for row in result.rows if row.template == template]
        if not group:
            continue
        axes[0].scatter(
            [row.hardware_cost for row in group],
            [row.refined_fidelity for row in group],
            label=template,
            alpha=0.75,
            color=color_by_template[template],
        )
        axes[1].scatter(
            [row.proposal_fidelity for row in group],
            [row.refined_fidelity for row in group],
            alpha=0.75,
            color=color_by_template[template],
        )
        axes[2].scatter(
            [row.n_cz for row in group],
            [row.regularized_score for row in group],
            alpha=0.75,
            color=color_by_template[template],
        )

    frontier = pareto_frontier_rows(result)
    if frontier:
        axes[0].scatter(
            [row.hardware_cost for row in frontier],
            [row.refined_fidelity for row in frontier],
            marker="*",
            s=140,
            color="black",
            label="Pareto frontier",
        )

    axes[0].set_title("Fidelity vs hardware cost")
    axes[0].set_xlabel("hardware cost proxy")
    axes[0].set_ylabel("refined unitary fidelity")
    axes[0].legend()

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("Refinement gain")
    axes[1].set_xlabel("proposal fidelity")
    axes[1].set_ylabel("refined unitary fidelity")

    axes[2].set_title("Regularized score")
    axes[2].set_xlabel("number of CZ gates")
    axes[2].set_ylabel("refined fidelity - cost")

    fig.suptitle(f"Pareto candidate sampling: {result.target.name}")
    fig.tight_layout()
    plt.show()
