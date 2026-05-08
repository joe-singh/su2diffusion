import torch

from su2diffusion import (
    ParetoCandidateRow,
    ParetoCircuitResult,
    ParetoScoringConfig,
    get_three_qubit_cz_template,
    make_hamiltonian_target,
    pareto_frontier_rows,
    pareto_hardware_cost,
    rescore_pareto_circuit_result,
    run_pareto_circuit_sampling,
    sample_haar,
    top_pareto_rows,
)


def _row(
    name: str,
    cost: float,
    fidelity: float,
    is_pareto: bool = False,
) -> ParetoCandidateRow:
    return ParetoCandidateRow(
        target="target",
        template=name,
        source="test",
        candidate_rank=1,
        n_cz=3,
        n_local_gates=12,
        proposal_fidelity=fidelity - 0.1,
        refined_fidelity=fidelity,
        steps_to_threshold=1,
        movement_mean=0.1,
        movement_max=0.2,
        local_angle_sum=1.0,
        hardware_cost=cost,
        regularized_score=fidelity - cost,
        is_pareto=is_pareto,
        slot_indices=tuple(range(12)),
        slot_labels=("test",) * 12,
        refined_gates=torch.zeros(12, 4),
    )


def test_pareto_hardware_cost_uses_all_terms() -> None:
    scoring = ParetoScoringConfig(
        cz_weight=0.1,
        local_gate_weight=0.01,
        movement_weight=0.5,
        angle_weight=0.05,
    )

    cost = pareto_hardware_cost(
        n_cz=4,
        n_local_gates=15,
        movement_mean=0.2,
        local_angle_sum=3.0,
        scoring=scoring,
    )

    assert cost == 0.1 * 4 + 0.01 * 15 + 0.5 * 0.2 + 0.05 * 3.0


def test_frontier_rows_drop_dominated_candidates() -> None:
    target = make_hamiltonian_target([("XII", 0.1)], time=0.2, n_qubits=3)
    rows = [
        _row("cheap-good", cost=0.1, fidelity=0.9, is_pareto=True),
        _row("expensive-better", cost=0.2, fidelity=0.95, is_pareto=True),
        _row("dominated", cost=0.2, fidelity=0.85, is_pareto=False),
    ]
    result = ParetoCircuitResult(
        target=target,
        templates=(get_three_qubit_cz_template("line-3cz-a"),),
        source="test",
        scoring=ParetoScoringConfig(),
        rows=rows,
        reports={},
        threshold=0.99,
    )

    frontier = pareto_frontier_rows(result)
    assert [row.template for row in frontier] == ["cheap-good", "expensive-better"]

    top = top_pareto_rows(result, max_rows=1)
    assert len(top) == 1
    assert top[0].template == "cheap-good"

    rescored = rescore_pareto_circuit_result(
        result,
        ParetoScoringConfig(cz_weight=1.0, local_gate_weight=0.0, movement_weight=0.0, angle_weight=0.0),
    )
    assert rescored.scoring.cz_weight == 1.0
    assert all(row.hardware_cost == row.n_cz for row in rescored.rows)


def test_run_pareto_circuit_sampling_smoke() -> None:
    target = make_hamiltonian_target([("XII", 0.2)], time=0.3, n_qubits=3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    gates = sample_haar(10, device="cpu", generator=generator)
    labels = [f"g{i}" for i in range(gates.shape[0])]

    result = run_pareto_circuit_sampling(
        target,
        generated_gates=gates,
        generated_labels=labels,
        templates=("line-3cz-a", "line-4cz"),
        n_random_candidates=6,
        top_k_per_template=1,
        refinement_steps=1,
        refinement_lr=0.02,
        seed=11,
        show_progress=False,
    )

    assert len(result.rows) == 2
    assert {row.template for row in result.rows} == {"line-3cz-a", "line-4cz"}
    assert all(0.0 <= row.proposal_fidelity <= 1.0 for row in result.rows)
    assert all(0.0 <= row.refined_fidelity <= 1.0 for row in result.rows)
    assert all(torch.isfinite(row.refined_gates).all() for row in result.rows)
    assert any(row.is_pareto for row in result.rows)
