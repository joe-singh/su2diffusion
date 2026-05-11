from pathlib import Path

import pytest
import torch

from su2diffusion import (
    CircuitDiversityCandidateRow,
    CircuitDiversityResult,
    ParetoCandidateRow,
    ParetoCircuitResult,
    ParetoScoringConfig,
    PaperBenchmarkSuiteResult,
    ThreeQubitTokenRepeatabilityResult,
    ThreeQubitTokenRepeatabilityRunRow,
    compare_circuit_diversity_coverage,
    compare_circuit_diversity_properties,
    compare_circuit_unitary_cross_fidelity,
    compare_multitarget_circuit_diversity_properties,
    get_paper_benchmark_config,
    get_three_qubit_cz_template,
    make_hamiltonian_target,
    paper_benchmark_rows,
    paper_benchmark_summary_rows,
    save_level3f_diversity_artifacts,
    save_level3g_property_artifacts,
    save_level3h_multitarget_artifacts,
    save_level3i_angle_steering_artifacts,
    save_level3j_pareto_artifacts,
    save_paper_benchmark_artifacts,
    summarize_angle_steering_artifacts,
)


def _fake_suite() -> PaperBenchmarkSuiteResult:
    repeatability = ThreeQubitTokenRepeatabilityResult(
        runs=[],
        rows=[
            ThreeQubitTokenRepeatabilityRunRow(
                run=0,
                source="token",
                n_targets=4,
                proposal_mean=0.9,
                refined_mean=0.99,
                refinement_success=1.0,
                median_steps=4.0,
                mean_movement=0.1,
                max_movement=0.2,
            ),
            ThreeQubitTokenRepeatabilityRunRow(
                run=1,
                source="token",
                n_targets=4,
                proposal_mean=0.8,
                refined_mean=0.97,
                refinement_success=0.5,
                median_steps=6.0,
                mean_movement=0.2,
                max_movement=0.3,
            ),
            ThreeQubitTokenRepeatabilityRunRow(
                run=0,
                source="haar",
                n_targets=4,
                proposal_mean=0.1,
                refined_mean=0.95,
                refinement_success=0.25,
                median_steps=10.0,
                mean_movement=0.5,
                max_movement=0.7,
            ),
        ],
        threshold=0.99,
        template=get_three_qubit_cz_template("line-4cz"),
    )
    return PaperBenchmarkSuiteResult(
        config=get_paper_benchmark_config("smoke"),
        repeatability=repeatability,
    )


def test_named_paper_benchmark_configs() -> None:
    smoke = get_paper_benchmark_config("smoke")
    quick = get_paper_benchmark_config("quick")
    full = get_paper_benchmark_config("level3")

    assert smoke.train_steps < quick.train_steps < full.train_steps
    assert smoke.sample_count < quick.sample_count < full.sample_count
    assert smoke.n_heldout_targets == 1
    assert smoke.train_target_count == 1
    assert quick.n_heldout_targets == 4
    assert quick.train_target_count == 4
    assert smoke.template == quick.template == full.template == "line-4cz"
    assert full.n_heldout_targets == 48
    with pytest.raises(ValueError):
        get_paper_benchmark_config("not-a-config")


def test_paper_benchmark_rows_and_summary() -> None:
    suite = _fake_suite()

    rows = paper_benchmark_rows(suite)
    assert len(rows) == 3
    assert rows[0]["source"] == "token"

    summary = paper_benchmark_summary_rows(suite)
    assert [row.source for row in summary] == ["token", "haar"]
    token = summary[0]
    assert token.runs == 2
    assert token.n_per_run == 4
    assert token.proposal_mean == pytest.approx(0.85)
    assert token.refined_mean == pytest.approx(0.98)
    assert token.success_mean == pytest.approx(0.75)
    assert token.max_movement == pytest.approx(0.3)


def test_save_paper_benchmark_artifacts(tmp_path: Path) -> None:
    suite = _fake_suite()

    paths = save_paper_benchmark_artifacts(suite, tmp_path)

    assert set(paths) == {"config", "rows", "summary", "figure"}
    for path in paths.values():
        assert path.exists()
    assert "token" in paths["rows"].read_text()
    assert "proposal_mean" in paths["summary"].read_text()


def _fake_diversity_result(source: str, offset: float = 0.0) -> CircuitDiversityResult:
    target = make_hamiltonian_target(
        [("XII", 0.05)],
        time=0.1,
        name="fake-target",
        n_qubits=3,
        device="cpu",
    )
    template = get_three_qubit_cz_template("line-4cz")
    base = torch.zeros(template.n_slots, 4)
    base[:, 0] = 1.0
    shifted = base.clone()
    if offset:
        shifted[:, 0] = torch.cos(torch.tensor(offset))
        shifted[:, 1] = torch.sin(torch.tensor(offset))
    rows = [
        CircuitDiversityCandidateRow(
            target=target.name,
            template=template.name,
            source=source,
            candidate_rank=rank + 1,
            proposal_fidelity=0.5 + 0.1 * rank,
            refined_fidelity=0.995 - 0.001 * rank,
            steps_to_threshold=rank,
            movement_mean=0.1 + 0.01 * rank,
            movement_max=0.2 + 0.01 * rank,
            nearest_proposal_distance=0.0,
            nearest_refined_distance=0.0,
            slot_indices=tuple(range(template.n_slots)),
            slot_labels=tuple(None for _ in range(template.n_slots)),
            start_gates=base,
            refined_gates=shifted,
        )
        for rank in range(2)
    ]
    pairwise = torch.zeros(len(rows), len(rows))
    return CircuitDiversityResult(
        target=target,
        template=template,
        source=source,
        report=None,  # type: ignore[arg-type]
        rows=rows,
        proposal_pairwise_distances=pairwise,
        refined_pairwise_distances=pairwise,
        threshold=0.99,
        cluster_radius=0.15,
    )


def test_save_level3_artifacts(tmp_path: Path) -> None:
    search = _fake_diversity_result("generated-search")
    token = _fake_diversity_result("token-diffusion", offset=0.02)
    haar = _fake_diversity_result("haar", offset=0.03)
    results = {
        "token-diffusion": token,
        "generated-search": search,
        "haar": haar,
    }
    coverage = compare_circuit_diversity_coverage(search, results)
    cross = compare_circuit_unitary_cross_fidelity(search, results)
    properties = compare_circuit_diversity_properties(results)
    multitarget = compare_multitarget_circuit_diversity_properties(
        {"fake-target": results},
        regimes={"fake-target": "test"},
    )

    level3f = save_level3f_diversity_artifacts(coverage, cross, tmp_path / "level3f")
    level3g = save_level3g_property_artifacts(properties, tmp_path / "level3g")
    level3h = save_level3h_multitarget_artifacts(multitarget, tmp_path / "level3h")

    assert "coverage_csv" in level3f
    assert "cluster_summary_csv" in level3g
    assert "summary_csv" in level3h
    for paths in (level3f, level3g, level3h):
        for path in paths.values():
            assert path.exists()

    assert "token-diffusion" in level3f["coverage_csv"].read_text()
    assert "local_angle_sum" in level3g["cluster_tests_csv"].read_text()
    assert "fake-target" in level3h["summary_csv"].read_text()


def test_save_angle_steering_artifacts(tmp_path: Path) -> None:
    low = _fake_diversity_result("low-angle-data", offset=0.01)
    high = _fake_diversity_result("high-angle-data", offset=0.05)
    diagnostics = {
        "fake-target": {
            "low-angle-data": low,
            "high-angle-data": high,
        }
    }
    rows = summarize_angle_steering_artifacts(
        diagnostics,
        {"low-angle-data": 20.0, "high-angle-data": 35.0},
        source_order=("low-angle-data", "high-angle-data"),
    )
    assert [row["source"] for row in rows] == ["low-angle-data", "high-angle-data"]

    paths = save_level3i_angle_steering_artifacts(
        diagnostics,
        {"low-angle-data": 20.0, "high-angle-data": 35.0},
        tmp_path,
        source_order=("low-angle-data", "high-angle-data"),
    )

    assert set(paths) == {"metadata", "summary_csv", "summary_tex", "angle_steering_png", "angle_steering_pdf"}
    assert "high-angle-data" in paths["summary_csv"].read_text()


def test_save_pareto_artifacts(tmp_path: Path) -> None:
    target = make_hamiltonian_target(
        [("XII", 0.05)],
        time=0.1,
        name="fake-pareto",
        n_qubits=3,
        device="cpu",
    )
    template = get_three_qubit_cz_template("line-4cz")
    gates = torch.zeros(template.n_slots, 4)
    gates[:, 0] = 1.0
    rows = [
        ParetoCandidateRow(
            target=target.name,
            template=template.name,
            source="generated-search",
            candidate_rank=1,
            n_cz=len(template.edges),
            n_local_gates=template.n_slots,
            proposal_fidelity=0.4,
            refined_fidelity=0.99,
            steps_to_threshold=4,
            movement_mean=0.1,
            movement_max=0.2,
            local_angle_sum=25.0,
            hardware_cost=0.1,
            regularized_score=0.89,
            is_pareto=True,
            slot_indices=tuple(range(template.n_slots)),
            slot_labels=tuple("I" for _ in range(template.n_slots)),
            refined_gates=gates,
        )
    ]
    result = ParetoCircuitResult(
        target=target,
        templates=(template,),
        source="generated-search",
        scoring=ParetoScoringConfig(),
        rows=rows,
        reports={},
        threshold=0.99,
    )

    paths = save_level3j_pareto_artifacts(result, tmp_path)

    assert set(paths) == {
        "metadata",
        "candidates_csv",
        "candidates_tex",
        "template_summary_csv",
        "template_summary_tex",
        "frontier_csv",
        "frontier_tex",
        "pareto_candidate_cloud_png",
        "pareto_candidate_cloud_pdf",
    }
    assert "line-4cz" in paths["template_summary_csv"].read_text()
    assert "generated-search" not in paths["candidates_csv"].read_text()
