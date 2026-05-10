import torch

from su2diffusion import (
    ParetoCandidateRow,
    ParetoCircuitResult,
    ParetoScoringConfig,
    compare_circuit_diversity_coverage,
    compare_circuit_diversity_properties,
    compare_circuit_unitary_cross_fidelity,
    compare_multitarget_circuit_diversity_properties,
    get_three_qubit_cz_template,
    make_hamiltonian_target,
    pareto_frontier_rows,
    pareto_hardware_cost,
    print_circuit_diversity_coverage_summary,
    print_circuit_diversity_property_summary,
    print_circuit_diversity_property_tests,
    print_circuit_diversity,
    print_circuit_diversity_summary,
    print_circuit_unitary_cross_fidelity_summary,
    print_multitarget_circuit_diversity_property_summary,
    rescore_pareto_circuit_result,
    run_circuit_diversity_diagnostic,
    run_pareto_circuit_sampling,
    sample_haar,
    summarize_circuit_diversity,
    summarize_circuit_diversity_coverage,
    summarize_circuit_diversity_properties,
    summarize_circuit_unitary_cross_fidelity,
    test_circuit_diversity_properties as run_circuit_diversity_property_tests,
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


def test_run_circuit_diversity_diagnostic_generated_smoke(capsys) -> None:
    target = make_hamiltonian_target([("XII", 0.2)], time=0.3, n_qubits=3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(13)
    gates = sample_haar(12, device="cpu", generator=generator)
    labels = [f"g{i}" for i in range(gates.shape[0])]

    result = run_circuit_diversity_diagnostic(
        target,
        generated_gates=gates,
        generated_labels=labels,
        template="line-3cz-a",
        n_random_candidates=8,
        n_selected=3,
        refinement_steps=1,
        refinement_lr=0.02,
        seed=17,
        show_progress=False,
    )
    summary = summarize_circuit_diversity(result)
    print_circuit_diversity(result, max_rows=2)
    print_circuit_diversity_summary(result)
    captured = capsys.readouterr().out

    assert len(result.rows) == 3
    assert result.proposal_pairwise_distances.shape == (3, 3)
    assert result.refined_pairwise_distances.shape == (3, 3)
    assert summary.n == 3
    assert summary.proposal_clusters >= 1
    assert summary.refined_clusters >= 1
    assert "nn proposal" in captured
    assert all(0.0 <= row.proposal_fidelity <= 1.0 for row in result.rows)
    assert all(0.0 <= row.refined_fidelity <= 1.0 for row in result.rows)


def test_run_circuit_diversity_diagnostic_stack_smoke() -> None:
    target = make_hamiltonian_target([("ZII", 0.1)], time=0.2, n_qubits=3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)
    template_slots = 12
    stacks = sample_haar(4 * template_slots, device="cpu", generator=generator).reshape(4, template_slots, 4)

    result = run_circuit_diversity_diagnostic(
        target,
        candidate_stacks=stacks,
        template="line-3cz-a",
        source="token",
        n_selected=2,
        refinement_steps=1,
        refinement_lr=0.02,
        show_progress=False,
    )

    assert len(result.rows) == 2
    assert result.source == "token"
    assert result.rows[0].slot_labels == ("token",) * template_slots
    assert torch.isfinite(result.proposal_pairwise_distances).all()
    assert torch.isfinite(result.refined_pairwise_distances).all()


def test_circuit_diversity_coverage_smoke(capsys) -> None:
    target = make_hamiltonian_target([("XII", 0.1)], time=0.2, n_qubits=3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(31)
    gates = sample_haar(12, device="cpu", generator=generator)
    labels = [f"g{i}" for i in range(gates.shape[0])]

    reference = run_circuit_diversity_diagnostic(
        target,
        generated_gates=gates,
        generated_labels=labels,
        template="line-3cz-a",
        n_random_candidates=8,
        n_selected=3,
        refinement_steps=1,
        refinement_lr=0.02,
        threshold=0.0,
        cluster_radius=0.5,
        seed=37,
        show_progress=False,
    )
    stacks = sample_haar(4 * reference.template.n_slots, device="cpu", generator=generator).reshape(
        4,
        reference.template.n_slots,
        4,
    )
    haar = run_circuit_diversity_diagnostic(
        target,
        candidate_stacks=stacks,
        template=reference.template,
        source="haar",
        n_selected=3,
        refinement_steps=1,
        refinement_lr=0.02,
        threshold=0.0,
        cluster_radius=0.5,
        show_progress=False,
    )

    coverage = compare_circuit_diversity_coverage(
        reference,
        {"generated-search": reference, "haar": haar},
        cluster_radius=0.5,
        success_threshold=0.0,
    )
    rows = summarize_circuit_diversity_coverage(coverage)
    print_circuit_diversity_coverage_summary(coverage)
    captured = capsys.readouterr().out

    assert coverage.reference_cluster_count >= 1
    assert {row.source for row in rows} == {"generated-search", "haar"}
    assert all(0.0 <= row.coverage_fraction <= 1.0 for row in rows)
    assert "coverage" in captured


def test_circuit_diversity_property_comparison_smoke(capsys) -> None:
    target = make_hamiltonian_target([("XII", 0.1)], time=0.2, n_qubits=3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(51)
    gates = sample_haar(12, device="cpu", generator=generator)
    labels = [f"g{i}" for i in range(gates.shape[0])]

    reference = run_circuit_diversity_diagnostic(
        target,
        generated_gates=gates,
        generated_labels=labels,
        template="line-3cz-a",
        n_random_candidates=8,
        n_selected=3,
        refinement_steps=1,
        refinement_lr=0.02,
        threshold=0.0,
        cluster_radius=0.5,
        seed=53,
        show_progress=False,
    )
    stacks = sample_haar(4 * reference.template.n_slots, device="cpu", generator=generator).reshape(
        4,
        reference.template.n_slots,
        4,
    )
    token = run_circuit_diversity_diagnostic(
        target,
        candidate_stacks=stacks,
        template=reference.template,
        source="token-diffusion",
        n_selected=3,
        refinement_steps=1,
        refinement_lr=0.02,
        threshold=0.0,
        cluster_radius=0.5,
        show_progress=False,
    )

    properties = compare_circuit_diversity_properties(
        {"token-diffusion": token, "generated-search": reference},
        cluster_radius=0.5,
        success_threshold=0.0,
    )
    summaries = summarize_circuit_diversity_properties(properties, scope="cluster")
    test_rows = run_circuit_diversity_property_tests(
        properties,
        scope="cluster",
        haar_source="generated-search",
    )
    print_circuit_diversity_property_summary(properties, scope="cluster")
    print_circuit_diversity_property_tests(properties, scope="cluster", haar_source="generated-search")
    captured = capsys.readouterr().out

    assert {row.source for row in summaries} == {"token-diffusion", "generated-search"}
    assert all(row.n >= 1 for row in summaries)
    assert all(torch.isfinite(torch.tensor(row.cost_mean)) for row in summaries)
    assert {row.metric for row in test_rows} >= {"within_template_cost", "movement_mean", "local_angle_sum"}
    assert "within-template cost" in captured
    assert "effect size" in captured


def test_multitarget_circuit_diversity_property_summary_smoke(capsys) -> None:
    target = make_hamiltonian_target([("XII", 0.1)], time=0.2, n_qubits=3, name="multi-smoke")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(151)
    gates = sample_haar(12, device="cpu", generator=generator)
    labels = [f"g{i}" for i in range(gates.shape[0])]

    reference = run_circuit_diversity_diagnostic(
        target,
        generated_gates=gates,
        generated_labels=labels,
        template="line-3cz-a",
        n_random_candidates=8,
        n_selected=3,
        refinement_steps=1,
        refinement_lr=0.02,
        threshold=0.0,
        cluster_radius=0.5,
        seed=153,
        show_progress=False,
    )
    stacks = sample_haar(4 * reference.template.n_slots, device="cpu", generator=generator).reshape(
        4,
        reference.template.n_slots,
        4,
    )
    token = run_circuit_diversity_diagnostic(
        target,
        candidate_stacks=stacks,
        template=reference.template,
        source="token-diffusion",
        n_selected=3,
        refinement_steps=1,
        refinement_lr=0.02,
        threshold=0.0,
        cluster_radius=0.5,
        show_progress=False,
    )

    result = compare_multitarget_circuit_diversity_properties(
        {
            target.name: {
                "token-diffusion": token,
                "generated-search": reference,
                "haar": reference,
            }
        },
        regimes={target.name: "smoke"},
        cluster_radius=0.5,
        success_threshold=0.0,
    )
    print_multitarget_circuit_diversity_property_summary(result)
    captured = capsys.readouterr().out

    assert len(result.rows) == 1
    assert result.rows[0].target == target.name
    assert result.rows[0].regime == "smoke"
    assert 0.0 <= result.rows[0].token_coverage_fraction <= 1.0
    assert torch.isfinite(torch.tensor(result.rows[0].search_minus_token_angle))
    assert "multi-target" in captured
    assert "search-token A" in captured


def test_circuit_unitary_cross_fidelity_smoke(capsys) -> None:
    target = make_hamiltonian_target([("XII", 0.1)], time=0.2, n_qubits=3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(41)
    gates = sample_haar(12, device="cpu", generator=generator)
    labels = [f"g{i}" for i in range(gates.shape[0])]

    reference = run_circuit_diversity_diagnostic(
        target,
        generated_gates=gates,
        generated_labels=labels,
        template="line-3cz-a",
        n_random_candidates=8,
        n_selected=3,
        refinement_steps=1,
        refinement_lr=0.02,
        threshold=0.0,
        seed=43,
        show_progress=False,
    )
    stacks = sample_haar(4 * reference.template.n_slots, device="cpu", generator=generator).reshape(
        4,
        reference.template.n_slots,
        4,
    )
    haar = run_circuit_diversity_diagnostic(
        target,
        candidate_stacks=stacks,
        template=reference.template,
        source="haar",
        n_selected=3,
        refinement_steps=1,
        refinement_lr=0.02,
        threshold=0.0,
        show_progress=False,
    )

    result = compare_circuit_unitary_cross_fidelity(
        reference,
        {"generated-search": reference, "haar": haar},
        success_threshold=0.0,
    )
    rows = summarize_circuit_unitary_cross_fidelity(result)
    print_circuit_unitary_cross_fidelity_summary(result)
    captured = capsys.readouterr().out

    assert {row.source for row in rows} == {"generated-search", "haar"}
    assert result.matrices["generated-search"].shape == (3, 3)
    assert result.matrices["haar"].shape == (3, 3)
    assert all(0.0 <= row.best_match_min <= row.best_match_max <= 1.0 for row in rows)
    assert "best mean/std" in captured
