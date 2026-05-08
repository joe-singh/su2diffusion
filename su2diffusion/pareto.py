from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import torch

from .hamiltonian import (
    HamiltonianTarget,
    ThreeQubitCZTemplate,
    compose_three_qubit_template_units,
    get_three_qubit_cz_template,
    refine_three_qubit_template_candidate,
    synthesize_three_qubit_template_random_report,
)
from .quaternion import q_normalize, su2_distance
from .synthesis import (
    RefinementResult,
    SynthesisCandidate,
    SynthesisReport,
    quaternion_to_unitary,
    unitary_fidelity_batch,
)


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


@dataclass(frozen=True)
class CircuitDiversityCandidateRow:
    target: str
    template: str
    source: str
    candidate_rank: int
    proposal_fidelity: float
    refined_fidelity: float
    steps_to_threshold: int
    movement_mean: float
    movement_max: float
    nearest_proposal_distance: float
    nearest_refined_distance: float
    slot_indices: tuple[int, ...]
    slot_labels: tuple[str | None, ...]
    start_gates: torch.Tensor
    refined_gates: torch.Tensor


@dataclass(frozen=True)
class CircuitDiversityResult:
    target: HamiltonianTarget
    template: ThreeQubitCZTemplate
    source: str
    report: SynthesisReport
    rows: list[CircuitDiversityCandidateRow]
    proposal_pairwise_distances: torch.Tensor
    refined_pairwise_distances: torch.Tensor
    threshold: float
    cluster_radius: float


@dataclass(frozen=True)
class CircuitDiversitySummary:
    n: int
    proposal_mean: float
    proposal_max: float
    refined_mean: float
    refined_max: float
    success_fraction: float
    proposal_pairwise_mean: float
    refined_pairwise_mean: float
    proposal_nearest_median: float
    refined_nearest_median: float
    collapse_ratio: float
    proposal_clusters: int
    refined_clusters: int


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


def _stack_pairwise_distances(stacks: torch.Tensor) -> torch.Tensor:
    stacks = q_normalize(stacks)
    if stacks.ndim != 3 or stacks.shape[-1] != 4:
        raise ValueError("stacks must have shape (n, n_slots, 4)")
    if stacks.shape[0] == 0:
        return torch.empty(0, 0, device=stacks.device)
    left = stacks[:, None, :, :]
    right = stacks[None, :, :, :]
    distances = torch.minimum(su2_distance(left, right), su2_distance(left, -right))
    return distances.mean(dim=-1).detach().cpu()


def _offdiag_values(pairwise: torch.Tensor) -> torch.Tensor:
    if pairwise.numel() == 0 or pairwise.shape[0] <= 1:
        return torch.empty(0, dtype=pairwise.dtype, device=pairwise.device)
    mask = ~torch.eye(pairwise.shape[0], dtype=torch.bool, device=pairwise.device)
    return pairwise[mask]


def _nearest_distances(pairwise: torch.Tensor) -> torch.Tensor:
    if pairwise.numel() == 0:
        return torch.empty(0, dtype=pairwise.dtype, device=pairwise.device)
    if pairwise.shape[0] == 1:
        return torch.zeros(1, dtype=pairwise.dtype, device=pairwise.device)
    masked = pairwise.clone()
    masked.fill_diagonal_(float("inf"))
    return masked.min(dim=1).values


def _greedy_cluster_count(pairwise: torch.Tensor, radius: float) -> int:
    if radius < 0:
        raise ValueError("cluster radius must be non-negative")
    n = pairwise.shape[0]
    remaining = set(range(n))
    clusters = 0
    while remaining:
        center = min(remaining)
        neighbors = {
            i
            for i in remaining
            if float(pairwise[center, i]) <= radius
        }
        remaining -= neighbors
        clusters += 1
    return clusters


def _make_stack_report(
    stacks: torch.Tensor,
    target: HamiltonianTarget,
    template: ThreeQubitCZTemplate,
    source: str,
    n_selected: int,
) -> SynthesisReport:
    if stacks.ndim != 3 or stacks.shape[-1] != 4:
        raise ValueError("candidate_stacks must have shape (n, n_slots, 4)")
    if stacks.shape[1] != template.n_slots:
        raise ValueError(f"Template {template.name!r} expects {template.n_slots} slots")
    if stacks.shape[0] == 0:
        raise ValueError("candidate_stacks must contain at least one stack")

    stacks = q_normalize(stacks)
    units = quaternion_to_unitary(stacks)
    unitaries = compose_three_qubit_template_units(units, template)
    fidelities = unitary_fidelity_batch(unitaries, target.unitary.to(device=stacks.device, dtype=torch.complex64))
    values, rows = torch.topk(fidelities, k=min(n_selected, fidelities.numel()))
    candidates = [
        SynthesisCandidate(
            target=target.name,
            template=f"{template.name}-{source}",
            entangler=template.name,
            fidelity=float(value),
            slot_indices=(int(row),) * template.n_slots,
            slot_labels=(source,) * template.n_slots,
        )
        for value, row in zip(values.tolist(), rows.tolist())
    ]
    return SynthesisReport(
        name=f"{target.name} {template.name} {source} diversity",
        mode=source,
        target=target.name,
        entangler=template.name,
        candidates=candidates,
        fidelities=tuple(float(value) for value in fidelities.detach().cpu().tolist()),
    )


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
        "line-2cz-a",
        "line-2cz-b",
        "line-3cz-a",
        "line-3cz-b",
        "line-4cz",
        "line-4cz-b",
        "line-5cz-a",
        "line-5cz-b",
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


def run_circuit_diversity_diagnostic(
    target: HamiltonianTarget,
    generated_gates: torch.Tensor | None = None,
    generated_labels: list[str | None] | None = None,
    candidate_stacks: torch.Tensor | None = None,
    template: str | ThreeQubitCZTemplate = "line-4cz",
    source: str = "generated-search",
    n_random_candidates: int = 10_000,
    n_selected: int = 40,
    refinement_steps: int = 60,
    refinement_lr: float = 0.05,
    threshold: float = 0.99,
    cluster_radius: float = 0.15,
    seed: int = 0,
    show_progress: bool = True,
) -> CircuitDiversityResult:
    """Measure whether many good candidates are genuinely distinct circuits.

    For generated local gates, the function first samples random fixed-template
    circuits and keeps the best ``n_selected`` proposals. For already-generated
    circuit stacks, it keeps the best ``n_selected`` stacks. It then refines each
    selected proposal and compares pairwise distances in the product manifold
    before and after refinement.
    """

    template_obj = _coerce_template(template)
    if target.unitary.shape != (2**template_obj.n_qubits, 2**template_obj.n_qubits):
        raise ValueError(f"target must be a {template_obj.n_qubits}-qubit Hamiltonian target")
    if (generated_gates is None) == (candidate_stacks is None):
        raise ValueError("Provide exactly one of generated_gates or candidate_stacks")
    if n_selected <= 0:
        raise ValueError("n_selected must be positive")
    if n_random_candidates <= 0:
        raise ValueError("n_random_candidates must be positive")
    if refinement_steps <= 0:
        raise ValueError("refinement_steps must be positive")
    if refinement_lr <= 0:
        raise ValueError("refinement_lr must be positive")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")
    if cluster_radius < 0.0:
        raise ValueError("cluster_radius must be non-negative")

    selected: list[tuple[SynthesisCandidate, torch.Tensor, torch.Tensor, SynthesisCandidate]] = []
    if candidate_stacks is not None:
        stacks = q_normalize(candidate_stacks)
        report = _make_stack_report(
            stacks,
            target=target,
            template=template_obj,
            source=source,
            n_selected=n_selected,
        )
        for candidate in report.candidates:
            stack_index = candidate.slot_indices[0]
            start_stack = q_normalize(stacks[stack_index]).detach()
            refinement_candidate = SynthesisCandidate(
                target=target.name,
                template=template_obj.name,
                entangler=template_obj.name,
                fidelity=float(candidate.fidelity),
                slot_indices=tuple(range(template_obj.n_slots)),
                slot_labels=(source,) * template_obj.n_slots,
            )
            selected.append((candidate, start_stack, start_stack, refinement_candidate))
    else:
        if generated_gates is None:
            raise ValueError("generated_gates must be provided")
        if generated_gates.shape[0] == 0:
            raise ValueError("generated_gates must contain at least one gate")
        if generated_labels is not None and len(generated_labels) != generated_gates.shape[0]:
            raise ValueError("generated_labels must have one entry per generated gate")
        gates = q_normalize(generated_gates)
        report = synthesize_three_qubit_template_random_report(
            gates,
            target_unitary=target.unitary,
            target_name=target.name,
            template=template_obj,
            n_candidates=n_random_candidates,
            top_k=n_selected,
            local_labels=generated_labels,
            seed=seed,
            name=f"{target.name} {template_obj.name} {source} diversity",
            mode=source,
            keep_fidelities=True,
        )
        for candidate in report.candidates:
            start_stack = _candidate_start_stack(gates, candidate)
            selected.append((candidate, start_stack, gates, candidate))

    iterator = selected
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc=f"Refining diversity candidates: {target.name}", dynamic_ncols=True)

    refinements: list[RefinementResult] = []
    for candidate, _, local_gates, refinement_candidate in iterator:
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(rank=len(refinements) + 1)
        refinements.append(
            refine_three_qubit_template_candidate(
                local_gates,
                refinement_candidate,
                target_unitary=target.unitary,
                template=template_obj,
                num_steps=refinement_steps,
                lr=refinement_lr,
            )
        )

    start_stacks = torch.stack([item[1] for item in selected]).detach()
    refined_stacks = torch.stack([item.refined_gates for item in refinements]).detach()
    proposal_pairwise = _stack_pairwise_distances(start_stacks)
    refined_pairwise = _stack_pairwise_distances(refined_stacks)
    proposal_nearest = _nearest_distances(proposal_pairwise)
    refined_nearest = _nearest_distances(refined_pairwise)

    rows: list[CircuitDiversityCandidateRow] = []
    for rank, ((candidate, start_stack, _, _), refinement) in enumerate(zip(selected, refinements), start=1):
        _, movement_mean, movement_max = _refinement_movement(start_stack, refinement.refined_gates)
        rows.append(
            CircuitDiversityCandidateRow(
                target=target.name,
                template=template_obj.name,
                source=source,
                candidate_rank=rank,
                proposal_fidelity=float(candidate.fidelity),
                refined_fidelity=float(refinement.refined_fidelity),
                steps_to_threshold=_steps_to_threshold(
                    refinement.initial_fidelity,
                    refinement.fidelity_trace,
                    threshold,
                ),
                movement_mean=movement_mean,
                movement_max=movement_max,
                nearest_proposal_distance=float(proposal_nearest[rank - 1].item()),
                nearest_refined_distance=float(refined_nearest[rank - 1].item()),
                slot_indices=candidate.slot_indices,
                slot_labels=candidate.slot_labels,
                start_gates=start_stack.detach(),
                refined_gates=refinement.refined_gates.detach(),
            )
        )

    return CircuitDiversityResult(
        target=target,
        template=template_obj,
        source=source,
        report=report,
        rows=rows,
        proposal_pairwise_distances=proposal_pairwise,
        refined_pairwise_distances=refined_pairwise,
        threshold=threshold,
        cluster_radius=cluster_radius,
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


def summarize_circuit_diversity(
    result: CircuitDiversityResult,
    cluster_radius: float | None = None,
) -> CircuitDiversitySummary:
    if not result.rows:
        raise ValueError("result must contain at least one diversity row")
    radius = result.cluster_radius if cluster_radius is None else cluster_radius
    proposal_pairwise = _offdiag_values(result.proposal_pairwise_distances)
    refined_pairwise = _offdiag_values(result.refined_pairwise_distances)
    proposal_nearest = _nearest_distances(result.proposal_pairwise_distances)
    refined_nearest = _nearest_distances(result.refined_pairwise_distances)
    proposal = torch.tensor([row.proposal_fidelity for row in result.rows], dtype=torch.float32)
    refined = torch.tensor([row.refined_fidelity for row in result.rows], dtype=torch.float32)
    success = (refined >= result.threshold).float()

    proposal_pairwise_mean = float(proposal_pairwise.mean().item()) if proposal_pairwise.numel() else 0.0
    refined_pairwise_mean = float(refined_pairwise.mean().item()) if refined_pairwise.numel() else 0.0
    collapse_ratio = refined_pairwise_mean / proposal_pairwise_mean if proposal_pairwise_mean > 0.0 else float("nan")

    return CircuitDiversitySummary(
        n=len(result.rows),
        proposal_mean=float(proposal.mean().item()),
        proposal_max=float(proposal.max().item()),
        refined_mean=float(refined.mean().item()),
        refined_max=float(refined.max().item()),
        success_fraction=float(success.mean().item()),
        proposal_pairwise_mean=proposal_pairwise_mean,
        refined_pairwise_mean=refined_pairwise_mean,
        proposal_nearest_median=float(proposal_nearest.median().item()) if proposal_nearest.numel() else 0.0,
        refined_nearest_median=float(refined_nearest.median().item()) if refined_nearest.numel() else 0.0,
        collapse_ratio=collapse_ratio,
        proposal_clusters=_greedy_cluster_count(result.proposal_pairwise_distances, radius),
        refined_clusters=_greedy_cluster_count(result.refined_pairwise_distances, radius),
    )


def print_circuit_diversity(result: CircuitDiversityResult, max_rows: int | None = 12) -> None:
    print(f"target:   {result.target.name}")
    print(f"template: {result.template.name}")
    print(f"source:   {result.source}")
    print(f"radius:   {result.cluster_radius:g}")
    print()
    header = "rank   proposal   refined   steps   move    nn proposal   nn refined"
    print(header)
    print("-" * len(header))
    rows = result.rows if max_rows is None else result.rows[:max_rows]
    for row in rows:
        steps = str(row.steps_to_threshold) if row.steps_to_threshold >= 0 else "miss"
        print(
            f"{row.candidate_rank:<4} "
            f"{row.proposal_fidelity:>9.4f}  "
            f"{row.refined_fidelity:>8.4f}  "
            f"{steps:>6}  "
            f"{row.movement_mean:>6.4f}    "
            f"{row.nearest_proposal_distance:>8.4f}     "
            f"{row.nearest_refined_distance:>8.4f}"
        )
    if max_rows is not None and len(result.rows) > max_rows:
        print(f"... {len(result.rows) - max_rows} more")


def print_circuit_diversity_summary(result: CircuitDiversityResult) -> None:
    summary = summarize_circuit_diversity(result)
    header = (
        "n   proposal mean/max   refined mean/max   success   pairwise before/after   "
        "nearest before/after   clusters before/after   collapse"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{summary.n:<3} "
        f"{summary.proposal_mean:>8.4f}/{summary.proposal_max:<8.4f} "
        f"{summary.refined_mean:>8.4f}/{summary.refined_max:<8.4f} "
        f"{100 * summary.success_fraction:>6.1f}%   "
        f"{summary.proposal_pairwise_mean:>8.4f}/{summary.refined_pairwise_mean:<8.4f} "
        f"{summary.proposal_nearest_median:>8.4f}/{summary.refined_nearest_median:<8.4f} "
        f"{summary.proposal_clusters:>8}/{summary.refined_clusters:<8} "
        f"{summary.collapse_ratio:>8.4f}"
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


def plot_circuit_diversity(result: CircuitDiversityResult) -> None:
    if not result.rows:
        raise ValueError("result must contain at least one diversity row")

    proposal_pairwise = _offdiag_values(result.proposal_pairwise_distances).numpy()
    refined_pairwise = _offdiag_values(result.refined_pairwise_distances).numpy()
    proposal_nearest = _nearest_distances(result.proposal_pairwise_distances).numpy()
    refined_nearest = _nearest_distances(result.refined_pairwise_distances).numpy()
    proposal = [row.proposal_fidelity for row in result.rows]
    refined = [row.refined_fidelity for row in result.rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if len(proposal_pairwise):
        axes[0].hist(proposal_pairwise, bins=24, alpha=0.6, density=True, label="proposal")
    if len(refined_pairwise):
        axes[0].hist(refined_pairwise, bins=24, alpha=0.6, density=True, label="refined")
    axes[0].axvline(result.cluster_radius, linestyle="--", color="black", linewidth=1, label="cluster radius")
    axes[0].set_xlabel("mean sign-invariant slotwise SU(2) distance")
    axes[0].set_ylabel("density")
    axes[0].set_title("Pairwise circuit distances")
    axes[0].legend()

    axes[1].scatter(proposal, refined, alpha=0.75)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[1].axhline(result.threshold, linestyle=":", color="black", linewidth=1)
    axes[1].set_xlabel("proposal fidelity")
    axes[1].set_ylabel("refined fidelity")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Refinement gain")

    axes[2].boxplot([proposal_nearest, refined_nearest], labels=["proposal", "refined"], showmeans=True)
    axes[2].axhline(result.cluster_radius, linestyle="--", color="black", linewidth=1)
    axes[2].set_ylabel("nearest-neighbor sign-invariant SU(2)^n distance")
    axes[2].set_title("Local crowding")

    fig.suptitle(f"Circuit diversity: {result.target.name} on {result.template.name}")
    fig.tight_layout()
    plt.show()
