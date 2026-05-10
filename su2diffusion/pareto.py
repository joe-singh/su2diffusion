from dataclasses import dataclass, replace
import math

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


@dataclass(frozen=True)
class CircuitDiversityCoverageResult:
    reference: CircuitDiversityResult
    results: dict[str, CircuitDiversityResult]
    cluster_radius: float
    success_threshold: float
    reference_cluster_count: int


@dataclass(frozen=True)
class CircuitDiversityCoverageSummary:
    source: str
    n: int
    n_success: int
    proposal_mean: float
    proposal_std: float
    refined_mean: float
    refined_std: float
    success_fraction: float
    median_steps: float
    pairwise_refined_mean: float
    within_cluster_mean: float
    across_cluster_mean: float
    cluster_count: int
    coverage_count: int
    coverage_fraction: float


@dataclass(frozen=True)
class CircuitUnitaryCrossFidelityResult:
    reference: CircuitDiversityResult
    results: dict[str, CircuitDiversityResult]
    success_threshold: float
    matrices: dict[str, torch.Tensor]


@dataclass(frozen=True)
class CircuitUnitaryCrossFidelitySummary:
    source: str
    n: int
    n_success: int
    best_match_mean: float
    best_match_std: float
    best_match_median: float
    best_match_min: float
    best_match_max: float
    fraction_above_099: float
    fraction_above_0999: float
    all_pairwise_mean: float


@dataclass(frozen=True)
class CircuitDiversityPropertyRow:
    source: str
    scope: str
    index: int
    n_members: int
    representative_rank: int
    proposal_fidelity: float
    refined_fidelity: float
    steps_to_threshold: int
    movement_mean: float
    movement_max: float
    local_angle_sum: float
    within_template_cost: float
    hardware_cost: float


@dataclass(frozen=True)
class CircuitDiversityPropertySummary:
    source: str
    scope: str
    n: int
    proposal_mean: float
    proposal_std: float
    refined_mean: float
    refined_std: float
    cost_mean: float
    cost_std: float
    cost_median: float
    cost_iqr: float
    angle_mean: float
    angle_std: float
    movement_mean: float
    movement_std: float
    max_movement_mean: float
    max_movement_std: float
    steps_median: float
    steps_iqr: float


@dataclass(frozen=True)
class CircuitDiversityPropertyTestRow:
    metric: str
    scope: str
    kruskal_h: float
    kruskal_p: float
    primary_vs_search_effect: float
    primary_vs_search_p: float
    primary_vs_search_p_bonf: float
    primary_vs_haar_effect: float
    primary_vs_haar_p: float
    primary_vs_haar_p_bonf: float


@dataclass(frozen=True)
class CircuitDiversityPropertyResult:
    results: dict[str, CircuitDiversityResult]
    cluster_radius: float
    success_threshold: float
    scoring: ParetoScoringConfig
    cluster_rows: list[CircuitDiversityPropertyRow]
    circuit_rows: list[CircuitDiversityPropertyRow]


@dataclass(frozen=True)
class CircuitDiversityMultiTargetPropertyRow:
    target: str
    regime: str
    reference_clusters: int
    token_success_fraction: float
    search_success_fraction: float
    haar_success_fraction: float
    token_clusters: int
    search_clusters: int
    haar_clusters: int
    token_coverage_count: int
    token_coverage_fraction: float
    token_angle_mean: float
    token_angle_std: float
    search_angle_mean: float
    search_angle_std: float
    haar_angle_mean: float
    haar_angle_std: float
    search_minus_token_angle: float
    token_search_angle_effect: float
    token_search_angle_p_bonf: float
    token_cost_mean: float
    search_cost_mean: float
    haar_cost_mean: float


@dataclass(frozen=True)
class CircuitDiversityMultiTargetPropertyResult:
    diagnostics: dict[str, dict[str, CircuitDiversityResult]]
    regimes: dict[str, str]
    coverages: dict[str, CircuitDiversityCoverageResult]
    properties: dict[str, CircuitDiversityPropertyResult]
    rows: list[CircuitDiversityMultiTargetPropertyRow]
    cluster_radius: float
    success_threshold: float
    primary_source: str
    reference_source: str
    search_source: str
    haar_source: str


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


def _stack_cross_distances(left_stacks: torch.Tensor, right_stacks: torch.Tensor) -> torch.Tensor:
    left_stacks = q_normalize(left_stacks)
    right_stacks = q_normalize(right_stacks)
    if left_stacks.ndim != 3 or left_stacks.shape[-1] != 4:
        raise ValueError("left_stacks must have shape (n, n_slots, 4)")
    if right_stacks.ndim != 3 or right_stacks.shape[-1] != 4:
        raise ValueError("right_stacks must have shape (n, n_slots, 4)")
    if left_stacks.shape[1] != right_stacks.shape[1]:
        raise ValueError("stacks must have the same number of local slots")
    if left_stacks.shape[0] == 0 or right_stacks.shape[0] == 0:
        return torch.empty(left_stacks.shape[0], right_stacks.shape[0], device=left_stacks.device)

    left = left_stacks[:, None, :, :]
    right = right_stacks[None, :, :, :]
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
    return len(_greedy_clusters(pairwise, radius))


def _greedy_clusters(pairwise: torch.Tensor, radius: float) -> tuple[tuple[int, ...], ...]:
    if radius < 0:
        raise ValueError("cluster radius must be non-negative")
    n = pairwise.shape[0]
    remaining = set(range(n))
    clusters: list[tuple[int, ...]] = []
    while remaining:
        center = min(remaining)
        neighbors = {
            i
            for i in remaining
            if float(pairwise[center, i]) <= radius
        }
        remaining -= neighbors
        clusters.append(tuple(sorted(neighbors)))
    return tuple(clusters)


def _successful_diversity_rows(
    result: CircuitDiversityResult,
    threshold: float,
) -> list[CircuitDiversityCandidateRow]:
    return [row for row in result.rows if row.refined_fidelity >= threshold]


def _row_stack(rows: list[CircuitDiversityCandidateRow], attr: str = "refined_gates") -> torch.Tensor:
    if not rows:
        return torch.empty(0, 0, 4)
    return torch.stack([getattr(row, attr).detach().cpu() for row in rows])


def _mean_std(values: torch.Tensor) -> tuple[float, float]:
    if values.numel() == 0:
        return float("nan"), float("nan")
    return float(values.mean().item()), float(values.std(unbiased=False).item())


def _finite_values(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().to(dtype=torch.float64).flatten()
    return values[torch.isfinite(values)]


def _median_iqr(values: torch.Tensor) -> tuple[float, float]:
    values = _finite_values(values)
    if values.numel() == 0:
        return float("nan"), float("nan")
    median = float(values.median().item())
    if values.numel() == 1:
        return median, 0.0
    q25, q75 = torch.quantile(values, torch.tensor([0.25, 0.75], dtype=values.dtype))
    return median, float((q75 - q25).item())


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().to(dtype=torch.float64).flatten()
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    n = values.numel()
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        average_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = average_rank
        i = j
    return ranks


def _chi_square_survival(value: float, df: int) -> float:
    if not math.isfinite(value) or value < 0.0:
        return float("nan")
    if df == 1:
        return math.erfc(math.sqrt(value / 2.0))
    if df == 2:
        return math.exp(-0.5 * value)
    return float("nan")


def _kruskal_wallis(groups: list[torch.Tensor]) -> tuple[float, float]:
    groups = [_finite_values(group) for group in groups if _finite_values(group).numel()]
    if len(groups) < 2:
        return float("nan"), float("nan")
    all_values = torch.cat(groups)
    n_total = all_values.numel()
    if n_total <= len(groups):
        return float("nan"), float("nan")
    ranks = _average_ranks(all_values)
    offset = 0
    rank_sum_term = 0.0
    for group in groups:
        n_group = group.numel()
        rank_sum = float(ranks[offset : offset + n_group].sum().item())
        rank_sum_term += rank_sum * rank_sum / n_group
        offset += n_group
    h_value = 12.0 / (n_total * (n_total + 1.0)) * rank_sum_term - 3.0 * (n_total + 1.0)
    h_value = max(0.0, h_value)
    return h_value, _chi_square_survival(h_value, len(groups) - 1)


def _mann_whitney_effect_p(primary: torch.Tensor, other: torch.Tensor) -> tuple[float, float]:
    primary = _finite_values(primary)
    other = _finite_values(other)
    n_primary = primary.numel()
    n_other = other.numel()
    if n_primary == 0 or n_other == 0:
        return float("nan"), float("nan")
    combined = torch.cat([primary, other])
    ranks = _average_ranks(combined)
    rank_sum_primary = float(ranks[:n_primary].sum().item())
    u_primary = rank_sum_primary - n_primary * (n_primary + 1.0) / 2.0
    effect = 2.0 * u_primary / (n_primary * n_other) - 1.0
    mean_u = n_primary * n_other / 2.0
    std_u = math.sqrt(n_primary * n_other * (n_primary + n_other + 1.0) / 12.0)
    if std_u == 0.0:
        return effect, float("nan")
    z_value = (u_primary - mean_u) / std_u
    p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return effect, p_value


def _property_values(
    rows: list[CircuitDiversityPropertyRow],
    metric: str,
) -> torch.Tensor:
    values = []
    for row in rows:
        value = getattr(row, metric)
        if metric == "steps_to_threshold" and value < 0:
            continue
        values.append(float(value))
    return torch.tensor(values, dtype=torch.float64)


def _property_source_rows(
    rows: list[CircuitDiversityPropertyRow],
    source: str,
) -> list[CircuitDiversityPropertyRow]:
    return [row for row in rows if row.source == source]


def _property_row_from_diversity_row(
    row: CircuitDiversityCandidateRow,
    *,
    source: str,
    scope: str,
    index: int,
    n_members: int,
    scoring: ParetoScoringConfig,
    template: ThreeQubitCZTemplate,
) -> CircuitDiversityPropertyRow:
    local_angle_sum = _local_angle_sum(row.refined_gates)
    within_template_cost = (
        scoring.movement_weight * row.movement_mean
        + scoring.angle_weight * local_angle_sum
    )
    hardware_cost = pareto_hardware_cost(
        n_cz=len(template.edges),
        n_local_gates=template.n_slots,
        movement_mean=row.movement_mean,
        local_angle_sum=local_angle_sum,
        scoring=scoring,
    )
    return CircuitDiversityPropertyRow(
        source=source,
        scope=scope,
        index=index,
        n_members=n_members,
        representative_rank=row.candidate_rank,
        proposal_fidelity=row.proposal_fidelity,
        refined_fidelity=row.refined_fidelity,
        steps_to_threshold=row.steps_to_threshold,
        movement_mean=row.movement_mean,
        movement_max=row.movement_max,
        local_angle_sum=local_angle_sum,
        within_template_cost=within_template_cost,
        hardware_cost=hardware_cost,
    )


def _cluster_within_across(pairwise: torch.Tensor, radius: float) -> tuple[int, float, float]:
    if pairwise.numel() == 0 or pairwise.shape[0] == 0:
        return 0, float("nan"), float("nan")
    clusters = _greedy_clusters(pairwise, radius)
    n = pairwise.shape[0]
    labels = torch.full((n,), -1, dtype=torch.long)
    for cluster_index, cluster in enumerate(clusters):
        labels[list(cluster)] = cluster_index

    offdiag = ~torch.eye(n, dtype=torch.bool)
    within_mask = (labels[:, None] == labels[None, :]) & offdiag
    across_mask = (labels[:, None] != labels[None, :]) & offdiag
    within = pairwise[within_mask]
    across = pairwise[across_mask]
    within_mean = float(within.mean().item()) if within.numel() else float("nan")
    across_mean = float(across.mean().item()) if across.numel() else float("nan")
    return len(clusters), within_mean, across_mean


def _unitaries_from_diversity_rows(
    rows: list[CircuitDiversityCandidateRow],
    template: ThreeQubitCZTemplate,
) -> torch.Tensor:
    dim = 2**template.n_qubits
    if not rows:
        return torch.empty(0, dim, dim, dtype=torch.complex64)
    stacks = _row_stack(rows)
    local_units = quaternion_to_unitary(stacks)
    return compose_three_qubit_template_units(local_units, template).detach().cpu()


def _unitary_cross_fidelity_matrix(
    source: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if source.ndim != 3 or reference.ndim != 3:
        raise ValueError("source and reference unitaries must have shape (n, dim, dim)")
    if source.shape[-2:] != reference.shape[-2:]:
        raise ValueError("source and reference unitaries must have matching dimensions")
    if source.shape[0] == 0 or reference.shape[0] == 0:
        return torch.empty(source.shape[0], reference.shape[0], dtype=torch.float32)
    dim = source.shape[-1]
    overlaps = torch.einsum("rij,sij->sr", reference.conj(), source).abs() / dim
    return overlaps.real.clamp(0.0, 1.0).detach().cpu()


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


def compare_circuit_diversity_coverage(
    reference: CircuitDiversityResult,
    results: dict[str, CircuitDiversityResult],
    cluster_radius: float | None = None,
    success_threshold: float | None = None,
) -> CircuitDiversityCoverageResult:
    """Compare successful samples against search-found reference solution clusters.

    The reference clusters are built from refined circuits that pass the success
    threshold. Each comparison source receives credit for a reference cluster if
    at least one of its successful refined circuits lies within ``cluster_radius``.
    """

    radius = reference.cluster_radius if cluster_radius is None else cluster_radius
    threshold = reference.threshold if success_threshold is None else success_threshold
    if radius < 0.0:
        raise ValueError("cluster_radius must be non-negative")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("success_threshold must be between 0 and 1")
    if not reference.rows:
        raise ValueError("reference must contain at least one diversity row")

    reference_rows = _successful_diversity_rows(reference, threshold)
    if not reference_rows:
        raise ValueError("reference must contain at least one successful row")
    reference_stacks = _row_stack(reference_rows)
    reference_pairwise = _stack_pairwise_distances(reference_stacks)
    reference_clusters = _greedy_clusters(reference_pairwise, radius)
    if not reference_clusters:
        raise ValueError("reference clustering produced no clusters")

    for name, result in results.items():
        if result.template.name != reference.template.name:
            raise ValueError(f"{name!r} uses template {result.template.name!r}, expected {reference.template.name!r}")
        if result.target.name != reference.target.name:
            raise ValueError(f"{name!r} uses target {result.target.name!r}, expected {reference.target.name!r}")

    return CircuitDiversityCoverageResult(
        reference=reference,
        results=dict(results),
        cluster_radius=radius,
        success_threshold=threshold,
        reference_cluster_count=len(reference_clusters),
    )


def summarize_circuit_diversity_coverage(
    coverage: CircuitDiversityCoverageResult,
) -> list[CircuitDiversityCoverageSummary]:
    reference_rows = _successful_diversity_rows(coverage.reference, coverage.success_threshold)
    reference_stacks = _row_stack(reference_rows)
    reference_pairwise = _stack_pairwise_distances(reference_stacks)
    reference_clusters = _greedy_clusters(reference_pairwise, coverage.cluster_radius)
    center_indices = [cluster[0] for cluster in reference_clusters]
    reference_centers = reference_stacks[center_indices]

    summaries: list[CircuitDiversityCoverageSummary] = []
    for source, result in coverage.results.items():
        rows = result.rows
        success_rows = _successful_diversity_rows(result, coverage.success_threshold)

        proposal_values = torch.tensor([row.proposal_fidelity for row in rows], dtype=torch.float32)
        refined_values = torch.tensor([row.refined_fidelity for row in rows], dtype=torch.float32)
        proposal_mean, proposal_std = _mean_std(proposal_values)
        refined_mean, refined_std = _mean_std(refined_values)
        success_fraction = len(success_rows) / len(rows) if rows else float("nan")

        reached_steps = torch.tensor(
            [row.steps_to_threshold for row in success_rows if row.steps_to_threshold >= 0],
            dtype=torch.float32,
        )
        median_steps = float(reached_steps.median().item()) if reached_steps.numel() else float("nan")

        if success_rows:
            stacks = _row_stack(success_rows)
            pairwise = _stack_pairwise_distances(stacks)
            offdiag = _offdiag_values(pairwise)
            pairwise_mean = float(offdiag.mean().item()) if offdiag.numel() else 0.0
            cluster_count, within_mean, across_mean = _cluster_within_across(pairwise, coverage.cluster_radius)
            cross = _stack_cross_distances(stacks, reference_centers)
            covered = (cross <= coverage.cluster_radius).any(dim=0)
            coverage_count = int(covered.sum().item())
        else:
            pairwise_mean = float("nan")
            cluster_count = 0
            within_mean = float("nan")
            across_mean = float("nan")
            coverage_count = 0

        coverage_fraction = (
            coverage_count / coverage.reference_cluster_count
            if coverage.reference_cluster_count
            else float("nan")
        )
        summaries.append(
            CircuitDiversityCoverageSummary(
                source=source,
                n=len(rows),
                n_success=len(success_rows),
                proposal_mean=proposal_mean,
                proposal_std=proposal_std,
                refined_mean=refined_mean,
                refined_std=refined_std,
                success_fraction=success_fraction,
                median_steps=median_steps,
                pairwise_refined_mean=pairwise_mean,
                within_cluster_mean=within_mean,
                across_cluster_mean=across_mean,
                cluster_count=cluster_count,
                coverage_count=coverage_count,
                coverage_fraction=coverage_fraction,
            )
        )
    return summaries


def print_circuit_diversity_coverage_summary(
    coverage: CircuitDiversityCoverageResult,
) -> None:
    print(f"reference: {coverage.reference.source}")
    print(f"target:    {coverage.reference.target.name}")
    print(f"template:  {coverage.reference.template.name}")
    print(f"success:   F >= {coverage.success_threshold:g}")
    print(f"radius:    {coverage.cluster_radius:g}")
    print(f"reference clusters: {coverage.reference_cluster_count}")
    print()
    header = (
        "source             n   success   proposal      refined       steps   "
        "pairwise   clusters   coverage   within/across"
    )
    print(header)
    print("-" * len(header))
    for row in summarize_circuit_diversity_coverage(coverage):
        steps = f"{row.median_steps:.1f}" if torch.isfinite(torch.tensor(row.median_steps)) else "nan"
        within = f"{row.within_cluster_mean:.4f}" if torch.isfinite(torch.tensor(row.within_cluster_mean)) else "nan"
        across = f"{row.across_cluster_mean:.4f}" if torch.isfinite(torch.tensor(row.across_cluster_mean)) else "nan"
        print(
            f"{row.source:<18} "
            f"{row.n:<3} "
            f"{100 * row.success_fraction:>6.1f}%   "
            f"{row.proposal_mean:>6.4f}/{row.proposal_std:<6.4f} "
            f"{row.refined_mean:>6.4f}/{row.refined_std:<6.4f} "
            f"{steps:>7}   "
            f"{row.pairwise_refined_mean:>7.4f}   "
            f"{row.cluster_count:>4}      "
            f"{row.coverage_count:>3}/{coverage.reference_cluster_count:<3} "
            f"{100 * row.coverage_fraction:>6.1f}%   "
            f"{within}/{across}"
        )


def compare_circuit_unitary_cross_fidelity(
    reference: CircuitDiversityResult,
    results: dict[str, CircuitDiversityResult],
    success_threshold: float | None = None,
) -> CircuitUnitaryCrossFidelityResult:
    """Compare full-unitary agreement between successful diversity samples.

    The reference set is usually the generated-search result. For each source,
    the returned matrix has shape ``(n_source_success, n_reference_success)`` and
    stores global-phase-invariant unitary fidelity between refined circuits.
    """

    threshold = reference.threshold if success_threshold is None else success_threshold
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("success_threshold must be between 0 and 1")
    reference_rows = _successful_diversity_rows(reference, threshold)
    if not reference_rows:
        raise ValueError("reference must contain at least one successful row")
    reference_units = _unitaries_from_diversity_rows(reference_rows, reference.template)

    matrices: dict[str, torch.Tensor] = {}
    for source, result in results.items():
        if result.template.name != reference.template.name:
            raise ValueError(f"{source!r} uses template {result.template.name!r}, expected {reference.template.name!r}")
        if result.target.name != reference.target.name:
            raise ValueError(f"{source!r} uses target {result.target.name!r}, expected {reference.target.name!r}")
        rows = _successful_diversity_rows(result, threshold)
        source_units = _unitaries_from_diversity_rows(rows, result.template)
        matrices[source] = _unitary_cross_fidelity_matrix(source_units, reference_units)

    return CircuitUnitaryCrossFidelityResult(
        reference=reference,
        results=dict(results),
        success_threshold=threshold,
        matrices=matrices,
    )


def summarize_circuit_unitary_cross_fidelity(
    result: CircuitUnitaryCrossFidelityResult,
) -> list[CircuitUnitaryCrossFidelitySummary]:
    summaries: list[CircuitUnitaryCrossFidelitySummary] = []
    for source, diversity in result.results.items():
        matrix = result.matrices[source]
        rows = diversity.rows
        n_success = matrix.shape[0]
        if matrix.numel():
            best = matrix.max(dim=1).values
            best_mean, best_std = _mean_std(best)
            summary = CircuitUnitaryCrossFidelitySummary(
                source=source,
                n=len(rows),
                n_success=n_success,
                best_match_mean=best_mean,
                best_match_std=best_std,
                best_match_median=float(best.median().item()),
                best_match_min=float(best.min().item()),
                best_match_max=float(best.max().item()),
                fraction_above_099=float((best >= 0.99).float().mean().item()),
                fraction_above_0999=float((best >= 0.999).float().mean().item()),
                all_pairwise_mean=float(matrix.mean().item()),
            )
        else:
            summary = CircuitUnitaryCrossFidelitySummary(
                source=source,
                n=len(rows),
                n_success=n_success,
                best_match_mean=float("nan"),
                best_match_std=float("nan"),
                best_match_median=float("nan"),
                best_match_min=float("nan"),
                best_match_max=float("nan"),
                fraction_above_099=float("nan"),
                fraction_above_0999=float("nan"),
                all_pairwise_mean=float("nan"),
            )
        summaries.append(summary)
    return summaries


def print_circuit_unitary_cross_fidelity_summary(
    result: CircuitUnitaryCrossFidelityResult,
) -> None:
    reference_success = result.matrices[next(iter(result.matrices))].shape[1] if result.matrices else 0
    print(f"reference: {result.reference.source}")
    print(f"target:    {result.reference.target.name}")
    print(f"template:  {result.reference.template.name}")
    print(f"success:   F >= {result.success_threshold:g}")
    print(f"reference successful circuits: {reference_success}")
    print()
    header = "source             success   best mean/std   median   min      max      >=0.99   >=0.999   all-pair mean"
    print(header)
    print("-" * len(header))
    for row in summarize_circuit_unitary_cross_fidelity(result):
        print(
            f"{row.source:<18} "
            f"{row.n_success:>3}/{row.n:<3}   "
            f"{row.best_match_mean:>6.4f}/{row.best_match_std:<6.4f} "
            f"{row.best_match_median:>7.4f} "
            f"{row.best_match_min:>7.4f} "
            f"{row.best_match_max:>7.4f} "
            f"{100 * row.fraction_above_099:>7.1f}% "
            f"{100 * row.fraction_above_0999:>8.1f}% "
            f"{row.all_pairwise_mean:>12.4f}"
        )


def compare_circuit_diversity_properties(
    results: dict[str, CircuitDiversityResult],
    cluster_radius: float | None = None,
    success_threshold: float | None = None,
    scoring: ParetoScoringConfig | None = None,
) -> CircuitDiversityPropertyResult:
    """Characterize successful decomposition families by simple circuit properties.

    The cluster-level rows use the highest-fidelity successful circuit in each
    slotwise SU(2)^n cluster as a representative. The per-circuit rows keep every
    successful circuit and are mainly a robustness check for the cluster-level
    comparison.
    """

    if not results:
        raise ValueError("results must contain at least one diversity result")
    first = next(iter(results.values()))
    radius = first.cluster_radius if cluster_radius is None else cluster_radius
    threshold = first.threshold if success_threshold is None else success_threshold
    scoring = scoring or ParetoScoringConfig()
    if radius < 0.0:
        raise ValueError("cluster_radius must be non-negative")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("success_threshold must be between 0 and 1")

    for source, result in results.items():
        if result.template.name != first.template.name:
            raise ValueError(f"{source!r} uses template {result.template.name!r}, expected {first.template.name!r}")
        if result.target.name != first.target.name:
            raise ValueError(f"{source!r} uses target {result.target.name!r}, expected {first.target.name!r}")

    cluster_rows: list[CircuitDiversityPropertyRow] = []
    circuit_rows: list[CircuitDiversityPropertyRow] = []
    for source, result in results.items():
        success_rows = _successful_diversity_rows(result, threshold)
        for index, row in enumerate(success_rows):
            circuit_rows.append(
                _property_row_from_diversity_row(
                    row,
                    source=source,
                    scope="circuit",
                    index=index,
                    n_members=1,
                    scoring=scoring,
                    template=result.template,
                )
            )

        if not success_rows:
            continue
        stacks = _row_stack(success_rows)
        pairwise = _stack_pairwise_distances(stacks)
        clusters = _greedy_clusters(pairwise, radius)
        for cluster_index, cluster in enumerate(clusters):
            representative_index = max(
                cluster,
                key=lambda item: success_rows[item].refined_fidelity,
            )
            cluster_rows.append(
                _property_row_from_diversity_row(
                    success_rows[representative_index],
                    source=source,
                    scope="cluster",
                    index=cluster_index,
                    n_members=len(cluster),
                    scoring=scoring,
                    template=result.template,
                )
            )

    return CircuitDiversityPropertyResult(
        results=dict(results),
        cluster_radius=radius,
        success_threshold=threshold,
        scoring=scoring,
        cluster_rows=cluster_rows,
        circuit_rows=circuit_rows,
    )


def summarize_circuit_diversity_properties(
    result: CircuitDiversityPropertyResult,
    scope: str = "cluster",
) -> list[CircuitDiversityPropertySummary]:
    if scope not in {"cluster", "circuit"}:
        raise ValueError("scope must be 'cluster' or 'circuit'")
    rows = result.cluster_rows if scope == "cluster" else result.circuit_rows
    summaries: list[CircuitDiversityPropertySummary] = []
    for source in result.results:
        source_rows = _property_source_rows(rows, source)
        proposal = _property_values(source_rows, "proposal_fidelity")
        refined = _property_values(source_rows, "refined_fidelity")
        cost = _property_values(source_rows, "within_template_cost")
        angle = _property_values(source_rows, "local_angle_sum")
        movement = _property_values(source_rows, "movement_mean")
        max_movement = _property_values(source_rows, "movement_max")
        steps = _property_values(source_rows, "steps_to_threshold")
        proposal_mean, proposal_std = _mean_std(_finite_values(proposal).to(dtype=torch.float32))
        refined_mean, refined_std = _mean_std(_finite_values(refined).to(dtype=torch.float32))
        cost_mean, cost_std = _mean_std(_finite_values(cost).to(dtype=torch.float32))
        angle_mean, angle_std = _mean_std(_finite_values(angle).to(dtype=torch.float32))
        movement_mean, movement_std = _mean_std(_finite_values(movement).to(dtype=torch.float32))
        max_movement_mean, max_movement_std = _mean_std(_finite_values(max_movement).to(dtype=torch.float32))
        cost_median, cost_iqr = _median_iqr(cost)
        steps_median, steps_iqr = _median_iqr(steps)
        summaries.append(
            CircuitDiversityPropertySummary(
                source=source,
                scope=scope,
                n=len(source_rows),
                proposal_mean=proposal_mean,
                proposal_std=proposal_std,
                refined_mean=refined_mean,
                refined_std=refined_std,
                cost_mean=cost_mean,
                cost_std=cost_std,
                cost_median=cost_median,
                cost_iqr=cost_iqr,
                angle_mean=angle_mean,
                angle_std=angle_std,
                movement_mean=movement_mean,
                movement_std=movement_std,
                max_movement_mean=max_movement_mean,
                max_movement_std=max_movement_std,
                steps_median=steps_median,
                steps_iqr=steps_iqr,
            )
        )
    return summaries


def test_circuit_diversity_properties(
    result: CircuitDiversityPropertyResult,
    scope: str = "cluster",
    primary_source: str = "token-diffusion",
    search_source: str = "generated-search",
    haar_source: str = "haar",
) -> list[CircuitDiversityPropertyTestRow]:
    """Run precommitted non-parametric comparisons for property diagnostics.

    The p-values use lightweight normal/chi-square approximations so the package
    does not need a SciPy dependency. Effect sizes are rank-biserial/Cliff-style:
    positive means the primary source tends to have larger values than the
    comparison source.
    """

    if scope not in {"cluster", "circuit"}:
        raise ValueError("scope must be 'cluster' or 'circuit'")
    rows = result.cluster_rows if scope == "cluster" else result.circuit_rows
    metrics = (
        "within_template_cost",
        "local_angle_sum",
        "movement_mean",
        "movement_max",
        "steps_to_threshold",
        "proposal_fidelity",
        "refined_fidelity",
    )
    test_rows: list[CircuitDiversityPropertyTestRow] = []
    for metric in metrics:
        groups = [
            _property_values(_property_source_rows(rows, source), metric)
            for source in result.results
        ]
        kruskal_h, kruskal_p = _kruskal_wallis(groups)
        primary = _property_values(_property_source_rows(rows, primary_source), metric)
        search = _property_values(_property_source_rows(rows, search_source), metric)
        haar = _property_values(_property_source_rows(rows, haar_source), metric)
        primary_search_effect, primary_search_p = _mann_whitney_effect_p(primary, search)
        primary_haar_effect, primary_haar_p = _mann_whitney_effect_p(primary, haar)
        test_rows.append(
            CircuitDiversityPropertyTestRow(
                metric=metric,
                scope=scope,
                kruskal_h=kruskal_h,
                kruskal_p=kruskal_p,
                primary_vs_search_effect=primary_search_effect,
                primary_vs_search_p=primary_search_p,
                primary_vs_search_p_bonf=min(1.0, 2.0 * primary_search_p)
                if math.isfinite(primary_search_p)
                else float("nan"),
                primary_vs_haar_effect=primary_haar_effect,
                primary_vs_haar_p=primary_haar_p,
                primary_vs_haar_p_bonf=min(1.0, 2.0 * primary_haar_p)
                if math.isfinite(primary_haar_p)
                else float("nan"),
            )
        )
    return test_rows


def print_circuit_diversity_property_summary(
    result: CircuitDiversityPropertyResult,
    scope: str = "cluster",
) -> None:
    label = "cluster representatives" if scope == "cluster" else "successful circuits"
    print(f"scope: {label}")
    print(f"radius: {result.cluster_radius:g}")
    print(f"success: F >= {result.success_threshold:g}")
    print("within-template cost: movement/angle terms only; fixed CZ/local-gate constants omitted")
    print()
    header = (
        "source             n   proposal      refined       cost mean/std   "
        "angle mean/std   move mean/std   max move   steps med/IQR"
    )
    print(header)
    print("-" * len(header))
    for row in summarize_circuit_diversity_properties(result, scope=scope):
        print(
            f"{row.source:<18} "
            f"{row.n:<3} "
            f"{row.proposal_mean:>6.4f}/{row.proposal_std:<6.4f} "
            f"{row.refined_mean:>6.4f}/{row.refined_std:<6.4f} "
            f"{row.cost_mean:>7.4f}/{row.cost_std:<7.4f} "
            f"{row.angle_mean:>7.3f}/{row.angle_std:<7.3f} "
            f"{row.movement_mean:>6.4f}/{row.movement_std:<6.4f} "
            f"{row.max_movement_mean:>7.4f}   "
            f"{row.steps_median:>5.1f}/{row.steps_iqr:<5.1f}"
        )


def print_circuit_diversity_property_tests(
    result: CircuitDiversityPropertyResult,
    scope: str = "cluster",
    primary_source: str = "token-diffusion",
    search_source: str = "generated-search",
    haar_source: str = "haar",
) -> None:
    print(f"scope: {scope}")
    print(f"primary source: {primary_source}")
    print("effect size: positive means primary tends larger")
    print()
    header = (
        "metric                 KW H     KW p     "
        f"effect vs {search_source:<16} p(bonf)   "
        f"effect vs {haar_source:<8} p(bonf)"
    )
    print(header)
    print("-" * len(header))
    for row in test_circuit_diversity_properties(
        result,
        scope=scope,
        primary_source=primary_source,
        search_source=search_source,
        haar_source=haar_source,
    ):
        print(
            f"{row.metric:<22} "
            f"{row.kruskal_h:>6.2f}  "
            f"{row.kruskal_p:>7.2g}   "
            f"{row.primary_vs_search_effect:>8.3f}          "
            f"{row.primary_vs_search_p_bonf:>7.2g}   "
            f"{row.primary_vs_haar_effect:>8.3f}    "
            f"{row.primary_vs_haar_p_bonf:>7.2g}"
        )


def _summarize_sources_without_reference_clusters(
    results: dict[str, CircuitDiversityResult],
    threshold: float,
    radius: float,
) -> dict[str, CircuitDiversityCoverageSummary]:
    summaries: dict[str, CircuitDiversityCoverageSummary] = {}
    for source, result in results.items():
        rows = result.rows
        success_rows = _successful_diversity_rows(result, threshold)
        proposal_values = torch.tensor([row.proposal_fidelity for row in rows], dtype=torch.float32)
        refined_values = torch.tensor([row.refined_fidelity for row in rows], dtype=torch.float32)
        proposal_mean, proposal_std = _mean_std(proposal_values)
        refined_mean, refined_std = _mean_std(refined_values)
        success_fraction = len(success_rows) / len(rows) if rows else float("nan")
        reached_steps = torch.tensor(
            [row.steps_to_threshold for row in success_rows if row.steps_to_threshold >= 0],
            dtype=torch.float32,
        )
        median_steps = float(reached_steps.median().item()) if reached_steps.numel() else float("nan")

        if success_rows:
            stacks = _row_stack(success_rows)
            pairwise = _stack_pairwise_distances(stacks)
            offdiag = _offdiag_values(pairwise)
            pairwise_mean = float(offdiag.mean().item()) if offdiag.numel() else 0.0
            cluster_count, within_mean, across_mean = _cluster_within_across(pairwise, radius)
        else:
            pairwise_mean = float("nan")
            cluster_count = 0
            within_mean = float("nan")
            across_mean = float("nan")

        summaries[source] = CircuitDiversityCoverageSummary(
            source=source,
            n=len(rows),
            n_success=len(success_rows),
            proposal_mean=proposal_mean,
            proposal_std=proposal_std,
            refined_mean=refined_mean,
            refined_std=refined_std,
            success_fraction=success_fraction,
            median_steps=median_steps,
            pairwise_refined_mean=pairwise_mean,
            within_cluster_mean=within_mean,
            across_cluster_mean=across_mean,
            cluster_count=cluster_count,
            coverage_count=0,
            coverage_fraction=float("nan"),
        )
    return summaries


def compare_multitarget_circuit_diversity_properties(
    diagnostics: dict[str, dict[str, CircuitDiversityResult]],
    regimes: dict[str, str] | None = None,
    reference_source: str = "generated-search",
    primary_source: str = "token-diffusion",
    search_source: str = "generated-search",
    haar_source: str = "haar",
    cluster_radius: float | None = None,
    success_threshold: float | None = None,
    scoring: ParetoScoringConfig | None = None,
) -> CircuitDiversityMultiTargetPropertyResult:
    """Summarize coverage and property tests across several Hamiltonian targets.

    Each target entry should contain the same source names used in the single-target
    diversity diagnostics, typically token-diffusion, generated-search, and Haar.
    The returned rows precommit to the angle comparison used in the paper draft:
    token diffusion is the primary source, generated-search is the reference
    decomposition family, and negative effects mean token diffusion has smaller
    total local angle.
    """

    if not diagnostics:
        raise ValueError("diagnostics must contain at least one target")
    regimes = dict(regimes or {})
    scoring = scoring or ParetoScoringConfig()

    coverages: dict[str, CircuitDiversityCoverageResult] = {}
    properties: dict[str, CircuitDiversityPropertyResult] = {}
    rows: list[CircuitDiversityMultiTargetPropertyRow] = []
    resolved_radius = float("nan")
    resolved_threshold = float("nan")

    for target_name, result_map in diagnostics.items():
        for source in (reference_source, primary_source, search_source, haar_source):
            if source not in result_map:
                raise ValueError(f"{target_name!r} is missing source {source!r}")

        reference = result_map[reference_source]
        radius = reference.cluster_radius if cluster_radius is None else cluster_radius
        threshold = reference.threshold if success_threshold is None else success_threshold
        if radius < 0.0:
            raise ValueError("cluster_radius must be non-negative")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("success_threshold must be between 0 and 1")
        for source, source_result in result_map.items():
            if source_result.template.name != reference.template.name:
                raise ValueError(
                    f"{source!r} uses template {source_result.template.name!r}, "
                    f"expected {reference.template.name!r}"
                )
            if source_result.target.name != reference.target.name:
                raise ValueError(
                    f"{source!r} uses target {source_result.target.name!r}, "
                    f"expected {reference.target.name!r}"
                )

        reference_rows = _successful_diversity_rows(reference, threshold)
        coverage: CircuitDiversityCoverageResult | None = None
        reference_cluster_count = 0
        if reference_rows:
            coverage = compare_circuit_diversity_coverage(
                reference,
                result_map,
                cluster_radius=radius,
                success_threshold=threshold,
            )
            coverages[target_name] = coverage
            coverage_by_source = {
                item.source: item
                for item in summarize_circuit_diversity_coverage(coverage)
            }
            reference_cluster_count = coverage.reference_cluster_count
        else:
            coverage_by_source = _summarize_sources_without_reference_clusters(
                result_map,
                threshold=threshold,
                radius=radius,
            )
        property_result = compare_circuit_diversity_properties(
            result_map,
            cluster_radius=radius,
            success_threshold=threshold,
            scoring=scoring,
        )
        properties[target_name] = property_result
        resolved_radius = radius
        resolved_threshold = threshold

        summary_by_source = {
            item.source: item
            for item in summarize_circuit_diversity_properties(property_result, scope="cluster")
        }
        tests_by_metric = {
            item.metric: item
            for item in test_circuit_diversity_properties(
                property_result,
                scope="cluster",
                primary_source=primary_source,
                search_source=search_source,
                haar_source=haar_source,
            )
        }
        angle_test = tests_by_metric["local_angle_sum"]
        token_coverage = coverage_by_source[primary_source]
        search_coverage = coverage_by_source[search_source]
        haar_coverage = coverage_by_source[haar_source]
        token_summary = summary_by_source[primary_source]
        search_summary = summary_by_source[search_source]
        haar_summary = summary_by_source[haar_source]
        rows.append(
            CircuitDiversityMultiTargetPropertyRow(
                target=target_name,
                regime=regimes.get(target_name, "unspecified"),
                reference_clusters=reference_cluster_count,
                token_success_fraction=token_coverage.success_fraction,
                search_success_fraction=search_coverage.success_fraction,
                haar_success_fraction=haar_coverage.success_fraction,
                token_clusters=token_coverage.cluster_count,
                search_clusters=search_coverage.cluster_count,
                haar_clusters=haar_coverage.cluster_count,
                token_coverage_count=token_coverage.coverage_count,
                token_coverage_fraction=token_coverage.coverage_fraction,
                token_angle_mean=token_summary.angle_mean,
                token_angle_std=token_summary.angle_std,
                search_angle_mean=search_summary.angle_mean,
                search_angle_std=search_summary.angle_std,
                haar_angle_mean=haar_summary.angle_mean,
                haar_angle_std=haar_summary.angle_std,
                search_minus_token_angle=search_summary.angle_mean - token_summary.angle_mean,
                token_search_angle_effect=angle_test.primary_vs_search_effect,
                token_search_angle_p_bonf=angle_test.primary_vs_search_p_bonf,
                token_cost_mean=token_summary.cost_mean,
                search_cost_mean=search_summary.cost_mean,
                haar_cost_mean=haar_summary.cost_mean,
            )
        )

    return CircuitDiversityMultiTargetPropertyResult(
        diagnostics={name: dict(result_map) for name, result_map in diagnostics.items()},
        regimes=regimes,
        coverages=coverages,
        properties=properties,
        rows=rows,
        cluster_radius=resolved_radius,
        success_threshold=resolved_threshold,
        primary_source=primary_source,
        reference_source=reference_source,
        search_source=search_source,
        haar_source=haar_source,
    )


def print_multitarget_circuit_diversity_property_summary(
    result: CircuitDiversityMultiTargetPropertyResult,
) -> None:
    print("multi-target diversity/property summary")
    print(f"radius: {result.cluster_radius:g}")
    print(f"success: F >= {result.success_threshold:g}")
    print(f"primary source: {result.primary_source}")
    print(f"reference source: {result.reference_source}")
    print("angle effect: negative means primary has smaller total local angle than generated-search")
    print()
    header = (
        "target                    regime                 "
        "succ tok/search/haar   clusters tok/search/haar   coverage      "
        "A token/search/haar        search-token A   effect    p(bonf)"
    )
    print(header)
    print("-" * len(header))
    for row in result.rows:
        print(
            f"{row.target:<25} "
            f"{row.regime:<22} "
            f"{100.0 * row.token_success_fraction:>5.1f}/"
            f"{100.0 * row.search_success_fraction:>5.1f}/"
            f"{100.0 * row.haar_success_fraction:<5.1f}   "
            f"{row.token_clusters:>4}/{row.search_clusters:<4}/{row.haar_clusters:<4}          "
            f"{row.token_coverage_count:>4}/{row.reference_clusters:<4} "
            f"{100.0 * row.token_coverage_fraction:>5.1f}%   "
            f"{row.token_angle_mean:>6.2f}/"
            f"{row.search_angle_mean:>6.2f}/"
            f"{row.haar_angle_mean:<6.2f}       "
            f"{row.search_minus_token_angle:>7.2f}   "
            f"{row.token_search_angle_effect:>7.3f}   "
            f"{row.token_search_angle_p_bonf:>7.2g}"
        )


def plot_multitarget_circuit_diversity_properties(
    result: CircuitDiversityMultiTargetPropertyResult,
) -> None:
    if not result.rows:
        raise ValueError("result contains no rows")
    labels = [row.target for row in result.rows]
    xs = torch.arange(len(labels), dtype=torch.float32)
    width = 0.24

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(xs - width, [row.token_angle_mean for row in result.rows], width, label="token")
    axes[0].bar(xs, [row.search_angle_mean for row in result.rows], width, label="search")
    axes[0].bar(xs + width, [row.haar_angle_mean for row in result.rows], width, label="haar")
    axes[0].set_title("total local angle")
    axes[0].set_ylabel("radians")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].legend()

    axes[1].bar(xs - width, [row.token_success_fraction for row in result.rows], width, label="token")
    axes[1].bar(xs, [row.search_success_fraction for row in result.rows], width, label="search")
    axes[1].bar(xs + width, [row.haar_success_fraction for row in result.rows], width, label="haar")
    axes[1].set_title("refined success")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("fraction with F >= threshold")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")

    axes[2].bar(xs, [row.token_coverage_fraction for row in result.rows])
    axes[2].set_title("token coverage of search clusters")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("fraction")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()


def plot_circuit_diversity_properties(
    result: CircuitDiversityPropertyResult,
    scope: str = "cluster",
) -> None:
    if scope not in {"cluster", "circuit"}:
        raise ValueError("scope must be 'cluster' or 'circuit'")
    rows = result.cluster_rows if scope == "cluster" else result.circuit_rows
    if not rows:
        raise ValueError("result contains no property rows for the requested scope")
    sources = [source for source in result.results if _property_source_rows(rows, source)]
    metrics = (
        ("within_template_cost", "within-template cost"),
        ("local_angle_sum", "total local angle"),
        ("movement_mean", "mean movement"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, (metric, title) in zip(axes[:3], metrics):
        data = [
            _finite_values(_property_values(_property_source_rows(rows, source), metric)).tolist()
            for source in sources
        ]
        axis.boxplot(data, labels=sources, showmeans=True)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, source in enumerate(sources):
        source_rows = _property_source_rows(rows, source)
        axes[3].scatter(
            [row.local_angle_sum for row in source_rows],
            [row.movement_mean for row in source_rows],
            label=source,
            alpha=0.75,
            color=colors[index % len(colors)],
        )
    axes[3].set_title("movement vs angle")
    axes[3].set_xlabel("total local angle")
    axes[3].set_ylabel("mean movement")
    axes[3].legend()
    fig.suptitle(f"Circuit diversity properties ({scope})")
    fig.tight_layout()


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


def plot_circuit_diversity_coverage(coverage: CircuitDiversityCoverageResult) -> None:
    rows = summarize_circuit_diversity_coverage(coverage)
    if not rows:
        raise ValueError("coverage must contain at least one source")

    labels = [row.source for row in rows]
    refined = [[row.refined_fidelity for row in coverage.results[label].rows] for label in labels]
    coverage_values = [row.coverage_fraction for row in rows]
    clusters = [row.cluster_count for row in rows]
    pairwise = [row.pairwise_refined_mean for row in rows]
    within = [row.within_cluster_mean for row in rows]
    across = [row.across_cluster_mean for row in rows]

    x = torch.arange(len(labels), dtype=torch.float32).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].boxplot(refined, labels=labels, showmeans=True)
    axes[0].axhline(coverage.success_threshold, linestyle="--", color="black", linewidth=1)
    axes[0].set_ylabel("refined unitary fidelity")
    axes[0].set_title("Refined proposal quality")

    axes[1].bar(x - 0.18, coverage_values, width=0.36, label="reference coverage")
    if coverage.reference_cluster_count:
        normalized_clusters = [item / coverage.reference_cluster_count for item in clusters]
    else:
        normalized_clusters = [float("nan") for _ in clusters]
    axes[1].bar(x + 0.18, normalized_clusters, width=0.36, label="own clusters / reference")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("fraction")
    axes[1].set_title("Mode coverage")
    axes[1].legend()

    axes[2].scatter(labels, pairwise, label="pairwise", s=45)
    axes[2].scatter(labels, within, label="within cluster", s=45)
    axes[2].scatter(labels, across, label="across cluster", s=45)
    axes[2].axhline(coverage.cluster_radius, linestyle="--", color="black", linewidth=1)
    axes[2].set_ylabel("sign-invariant SU(2)^n distance")
    axes[2].set_title("Successful-sample diversity")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].legend()

    fig.suptitle(f"Circuit diversity coverage: {coverage.reference.target.name}")
    fig.tight_layout()
    plt.show()


def plot_circuit_unitary_cross_fidelity(result: CircuitUnitaryCrossFidelityResult) -> None:
    summaries = summarize_circuit_unitary_cross_fidelity(result)
    if not summaries:
        raise ValueError("result must contain at least one source")

    labels = [row.source for row in summaries]
    best_matches = []
    all_pairs = []
    for label in labels:
        matrix = result.matrices[label]
        if matrix.numel():
            best_matches.append(matrix.max(dim=1).values.numpy())
            all_pairs.append(matrix.flatten().numpy())
        else:
            best_matches.append([])
            all_pairs.append([])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(best_matches, labels=labels, showmeans=True)
    axes[0].axhline(0.99, linestyle="--", color="black", linewidth=1, label="0.99")
    axes[0].axhline(0.999, linestyle=":", color="black", linewidth=1, label="0.999")
    axes[0].set_ylabel("best unitary fidelity to reference set")
    axes[0].set_title("Best cross-method match")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()

    bins = torch.linspace(0.0, 1.0, 41).numpy()
    for label, values in zip(labels, all_pairs):
        if len(values):
            axes[1].hist(values, bins=bins, alpha=0.45, density=True, label=label)
    axes[1].axvline(0.99, linestyle="--", color="black", linewidth=1)
    axes[1].set_xlabel("unitary fidelity to reference circuits")
    axes[1].set_ylabel("density")
    axes[1].set_title("All successful cross-pairs")
    axes[1].legend()

    fig.suptitle(f"Unitary cross-fidelity: {result.reference.target.name}")
    fig.tight_layout()
    plt.show()
