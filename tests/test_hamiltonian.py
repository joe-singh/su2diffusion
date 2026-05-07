import torch

from su2diffusion.circuit import CircuitExperimentConfig, CircuitTrainConfig
from su2diffusion.data import center_names_for_config, centers_for_config, DataConfig
from su2diffusion.diffusion import DiffusionSchedule
from su2diffusion.hamiltonian import (
    HamiltonianConditionedDiffusionResult,
    HamiltonianDenoiseAblationResult,
    HamiltonianDenoiseDiagnosticResult,
    HamiltonianDenoiseNormalizationResult,
    HamiltonianSkeletonDenoiseComparisonResult,
    HamiltonianSlotwiseDenoiseComparisonResult,
    HamiltonianTokenDenoiseComparisonResult,
    HamiltonianTokenDataScaleResult,
    HamiltonianTokenStackDataScaleResult,
    HamiltonianTokenStackTrainingBudgetResult,
    ThreeQubitTokenRefinementResult,
    HamiltonianTokenTemplateComparisonResult,
    HamiltonianTokenTrainingBudgetResult,
    HamiltonianTokenRepeatabilityResult,
    HamiltonianRepeatabilityRefinementResult,
    HamiltonianLevel1HeadlineResult,
    HamiltonianTemplateComparisonResult,
    HamiltonianConditionedOverfitDiagnosticResult,
    HamiltonianStackPredictor,
    HamiltonianPriorTrainConfig,
    HamiltonianSupervisedTrainConfig,
    compose_three_qubit_template_units,
    cz_on_qubits,
    estimate_hamiltonian_denoise_target_scale,
    evaluate_hamiltonian_conditioned_denoising,
    evaluate_hamiltonian_skeleton_conditioned_denoising,
    evaluate_hamiltonian_stack_predictor,
    generate_three_qubit_hamiltonian_solution_dataset,
    generate_hamiltonian_solution_dataset,
    hamiltonian_denoise_diagnostic_from_model,
    hamiltonian_from_terms,
    hamiltonian_target_features,
    make_hamiltonian_target,
    make_random_pauli_hamiltonian_targets,
    parse_pauli_string,
    pauli_string_matrix,
    print_hamiltonian_budget_refinement_summary,
    print_hamiltonian_conditioned_diffusion,
    print_hamiltonian_conditioned_diffusion_summary,
    print_hamiltonian_denoise_ablation,
    print_hamiltonian_denoise_diagnostic,
    print_hamiltonian_denoise_normalization,
    print_hamiltonian_skeleton_denoise_comparison,
    print_hamiltonian_slotwise_denoise_comparison,
    print_hamiltonian_token_conditioned_diffusion_summary,
    print_hamiltonian_token_data_scale_summary,
    print_hamiltonian_token_denoise_comparison,
    print_hamiltonian_token_heldout_comparison_summary,
    print_hamiltonian_token_repeatability_summary,
    print_hamiltonian_token_stack_data_scale_summary,
    print_hamiltonian_token_stack_training_budget_summary,
    print_hamiltonian_token_template_comparison,
    print_hamiltonian_token_training_budget_summary,
    print_hamiltonian_repeatability_refinement,
    print_hamiltonian_repeatability_refinement_summary,
    print_hamiltonian_level1_headline_table,
    print_hamiltonian_conditioned_overfit_diagnostic,
    print_hamiltonian_conditioned_overfit_summary,
    print_hamiltonian_mixture_refinement_summary,
    print_hamiltonian_prior_mixture_summary,
    print_hamiltonian_prior_search,
    print_hamiltonian_prior_search_summary,
    print_hamiltonian_target,
    print_hamiltonian_solution_dataset,
    print_hamiltonian_solution_dataset_summary,
    print_hamiltonian_template_comparison,
    print_hamiltonian_seed_ablation,
    print_hamiltonian_seed_ablation_summary,
    print_hamiltonian_supervised_summary,
    print_hamiltonian_supervised_split_summary,
    print_hamiltonian_suite,
    print_hamiltonian_suite_summary,
    print_hamiltonian_two_entangler_benchmark,
    print_hamiltonian_two_entangler_summary,
    print_three_qubit_template_summary,
    print_three_qubit_token_refinement_summary,
    run_hamiltonian_conditioned_diffusion_benchmark,
    run_hamiltonian_conditioned_denoise_diagnostic,
    run_hamiltonian_conditioned_overfit_diagnostic,
    run_hamiltonian_denoise_ablation,
    run_hamiltonian_denoise_normalization_comparison,
    run_hamiltonian_skeleton_denoise_comparison,
    run_hamiltonian_skeleton_denoise_diagnostic,
    run_hamiltonian_slotwise_denoise_comparison,
    run_hamiltonian_slotwise_denoise_diagnostic,
    run_hamiltonian_token_conditioned_diffusion_benchmark,
    run_hamiltonian_token_conditioned_overfit_diagnostic,
    run_hamiltonian_token_data_scale_benchmark,
    run_hamiltonian_token_denoise_comparison,
    run_hamiltonian_token_denoise_diagnostic,
    run_hamiltonian_token_repeatability_benchmark,
    run_hamiltonian_token_stack_data_scale_benchmark,
    run_hamiltonian_token_stack_training_budget_benchmark,
    run_hamiltonian_token_template_comparison,
    run_hamiltonian_token_training_budget_benchmark,
    run_three_qubit_hamiltonian_token_training_budget_benchmark,
    run_three_qubit_hamiltonian_token_refinement_benchmark,
    run_hamiltonian_repeatability_refinement_benchmark,
    refine_hamiltonian_prior_mixture,
    refine_hamiltonian_prior_mixture_budget_sweep,
    run_hamiltonian_prior_mixture_sweep,
    run_hamiltonian_prior_search_benchmark,
    run_hamiltonian_seed_ablation,
    run_hamiltonian_supervised_baseline,
    run_hamiltonian_supervised_split_baseline,
    run_hamiltonian_suite_benchmark,
    run_hamiltonian_template_comparison,
    run_hamiltonian_two_entangler_benchmark,
    run_three_qubit_template_benchmark,
    sample_hamiltonian_conditioned_circuit_reverse,
    synthesize_three_qubit_template_stack_report,
    summarize_hamiltonian_denoise_diagnostic,
    summarize_hamiltonian_denoise_ablation,
    summarize_hamiltonian_denoise_normalization,
    summarize_hamiltonian_skeleton_denoise_comparison,
    summarize_hamiltonian_slotwise_denoise_comparison,
    summarize_hamiltonian_token_conditioned_diffusion,
    summarize_hamiltonian_token_data_scale,
    summarize_hamiltonian_token_denoise_comparison,
    summarize_hamiltonian_token_heldout_comparison,
    summarize_hamiltonian_token_repeatability,
    summarize_hamiltonian_token_stack_data_scale,
    summarize_hamiltonian_token_stack_training_budget,
    summarize_hamiltonian_token_template_comparison,
    summarize_hamiltonian_token_training_budget,
    summarize_hamiltonian_repeatability_refinement,
    summarize_hamiltonian_level1_headline,
    summarize_hamiltonian_conditioned_diffusion,
    summarize_hamiltonian_conditioned_overfit_diagnostic,
    summarize_hamiltonian_suite,
    train_hamiltonian_conditioned_circuit_diffusion,
    train_hamiltonian_skeleton_conditioned_circuit_diffusion,
    train_hamiltonian_slotwise_circuit_diffusion,
    train_hamiltonian_token_circuit_diffusion,
    train_hamiltonian_slot_prior,
    unitary_from_hamiltonian,
)


def test_parse_pauli_string_accepts_compact_and_subscript_notation():
    assert parse_pauli_string("XI", n_qubits=2) == ("X", "I")
    assert parse_pauli_string("X0", n_qubits=2) == ("X", "I")
    assert parse_pauli_string("Z1", n_qubits=2) == ("I", "Z")
    assert parse_pauli_string("X0 Z1", n_qubits=2) == ("X", "Z")
    assert parse_pauli_string("X0Z1", n_qubits=2) == ("X", "Z")


def test_pauli_string_matrix_matches_tensor_product_convention():
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    identity = torch.eye(2, dtype=torch.complex64)

    assert torch.allclose(pauli_string_matrix("X0", n_qubits=2), torch.kron(x, identity))
    assert torch.allclose(pauli_string_matrix("Z1", n_qubits=2), torch.kron(identity, z))
    assert torch.allclose(pauli_string_matrix("X0Z1", n_qubits=2), torch.kron(x, z))


def test_hamiltonian_from_terms_is_hermitian_and_unitary_is_unitary():
    hamiltonian = hamiltonian_from_terms(
        [
            ("X0", 0.3),
            ("Z1", -0.2),
            ("X0X1", 0.15),
            ("ZZ", 0.05),
        ],
        n_qubits=2,
    )
    unitary = unitary_from_hamiltonian(hamiltonian, time=0.7)
    identity = torch.eye(4, dtype=torch.complex64)

    assert torch.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-6)
    assert torch.allclose(unitary.conj().T @ unitary, identity, atol=1e-5)


def test_hamiltonian_target_and_benchmark_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    target = make_hamiltonian_target(
        [
            ("XI", 0.25),
            ("IZ", -0.15),
            ("XX", 0.12),
        ],
        time=0.8,
        name="smoke-hamiltonian",
    )

    benchmark = run_hamiltonian_two_entangler_benchmark(
        target,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        keep_fidelities=False,
        seed=3,
    )
    print_hamiltonian_target(target)
    print_hamiltonian_two_entangler_benchmark(benchmark)
    print_hamiltonian_two_entangler_summary(benchmark)

    captured = capsys.readouterr().out
    assert "smoke-hamiltonian" in captured
    assert "generated random" in captured
    assert benchmark.generated_report.candidates[0].fidelity >= 0.0


def test_random_hamiltonian_suite_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=3,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.2,
        time=0.5,
        seed=4,
    )

    result = run_hamiltonian_suite_benchmark(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        keep_fidelities=False,
        seed=5,
    )
    rows = summarize_hamiltonian_suite(result)
    print_hamiltonian_suite(result)
    print_hamiltonian_suite_summary(result)

    captured = capsys.readouterr().out
    assert "target" in captured
    assert len(result.benchmarks) == 3
    assert [row.n_targets for row in rows] == [3, 3, 3, 3]
    assert all(0.0 <= row.mean_best <= 1.0 for row in rows)


def test_three_qubit_pauli_targets_and_template_composition():
    target = make_hamiltonian_target(
        [
            ("XII", 0.12),
            ("IZI", -0.08),
            ("IZZ", 0.05),
        ],
        time=0.4,
        name="three-qubit",
        n_qubits=3,
    )
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XII", "IZI", "IZZ"),
        coefficient_scale=0.1,
        time=0.3,
        n_qubits=3,
        seed=41,
    )
    identity_units = torch.eye(2, dtype=torch.complex64).expand(12, 2, 2).clone()
    composed = compose_three_qubit_template_units(identity_units, "line-3cz-a")
    expected = cz_on_qubits(3, (0, 1)) @ cz_on_qubits(3, (1, 2)) @ cz_on_qubits(3, (0, 1))

    assert target.unitary.shape == (8, 8)
    assert all(item.unitary.shape == (8, 8) for item in targets)
    assert torch.allclose(composed, expected, atol=1e-6)


def test_three_qubit_template_benchmark_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XII", "IZI", "IZZ"),
        coefficient_scale=0.1,
        time=0.3,
        n_qubits=3,
        seed=42,
    )

    result = run_three_qubit_template_benchmark(
        targets,
        generated_gates=centers,
        generated_labels=labels,
        templates=("line-3cz-a",),
        sources=("generated", "haar"),
        n_random_candidates=4,
        n_haar_gates=4,
        top_k=1,
        refinement_steps=1,
        refinement_lr=0.02,
        threshold=0.5,
        seed=43,
        show_progress=False,
    )
    print_three_qubit_template_summary(result)

    captured = capsys.readouterr().out
    assert "line-3cz-a" in captured
    assert len(result.rows) == 2
    assert set(result.reports) == {("line-3cz-a", "generated"), ("line-3cz-a", "haar")}
    assert all(row.n_targets == 2 for row in result.rows)
    assert all(0.0 <= row.proposal_mean <= 1.0 for row in result.rows)
    assert all(0.0 <= row.refined_mean <= 1.0 for row in result.rows)


def test_three_qubit_hamiltonian_solution_dataset_and_stack_report_smoke():
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=1,
        terms=("XII", "IZI", "IZZ"),
        coefficient_scale=0.1,
        time=0.3,
        n_qubits=3,
        seed=44,
    )

    dataset = generate_three_qubit_hamiltonian_solution_dataset(
        targets,
        generated_gates=centers,
        generated_labels=labels,
        template="line-4cz",
        n_random_candidates=4,
        top_k=1,
        seed=45,
        refinement_steps=1,
        refinement_lr=0.02,
        show_progress=False,
    )
    report = synthesize_three_qubit_template_stack_report(
        dataset.stacks,
        target_unitary=targets[0].unitary,
        target_name=targets[0].name,
        template="line-4cz",
        top_k=1,
    )

    assert dataset.stacks.shape == (1, 15, 4)
    assert dataset.refined_fidelities.shape == (1,)
    assert len(report.candidates) == 1
    assert 0.0 <= report.candidates[0].fidelity <= 1.0001


def test_three_qubit_hamiltonian_token_training_budget_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=1,
        terms=("XII", "IZZ"),
        coefficient_scale=0.1,
        time=0.3,
        n_qubits=3,
        seed=46,
    )
    config = CircuitExperimentConfig(
        name="test-three-qubit-token-budget",
        schedule=DiffusionSchedule(T=3, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=2, num_steps=1, hidden=16, n_terms=4),
        sample_count=2,
        eta=0.0,
        n_slots=15,
    )

    result = run_three_qubit_hamiltonian_token_training_budget_benchmark(
        train_target_counts=(1,),
        train_step_counts=(1,),
        heldout_targets=heldout_targets,
        generated_gates=centers,
        generated_labels=labels,
        config=config,
        template="line-4cz",
        terms=("XII", "IZZ"),
        coefficient_scale=0.1,
        time=0.3,
        train_seed=47,
        dataset_seed=48,
        heldout_baseline_seed=49,
        n_random_candidates=4,
        top_k=1,
        refinement_steps=1,
        refinement_lr=0.02,
        device="cpu",
        show_progress=False,
    )
    rows = summarize_hamiltonian_token_stack_training_budget(result)
    print_hamiltonian_token_stack_training_budget_summary(result)

    captured = capsys.readouterr().out
    assert "steps" in captured
    assert isinstance(result, HamiltonianTokenStackTrainingBudgetResult)
    assert result.train_dataset.stacks.shape[1] == 15
    assert set(result.diagnostics) == {(1, 1)}
    assert len(rows) == 1
    assert rows[0].n_solution_stacks == 1
    assert 0.0 <= rows[0].heldout_mean_best <= 1.0001

    refinement = run_three_qubit_hamiltonian_token_refinement_benchmark(
        result,
        generated_gates=centers,
        template="line-4cz",
        refinement_steps=1,
        refinement_lr=0.02,
        include_haar=True,
        show_progress=False,
    )
    print_three_qubit_token_refinement_summary(refinement)

    captured = capsys.readouterr().out
    assert "generated-search" in captured
    assert isinstance(refinement, ThreeQubitTokenRefinementResult)
    assert refinement.template.name == "line-4cz"
    assert len(refinement.rows) == 3
    assert {row.source for row in refinement.rows} == {"token", "generated-search", "haar"}
    assert all(0.0 <= row.refined_fidelity <= 1.0001 for row in refinement.rows)


def test_hamiltonian_solution_dataset_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=7,
    )

    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=8,
    )
    print_hamiltonian_solution_dataset(dataset)
    print_hamiltonian_solution_dataset_summary(dataset)

    captured = capsys.readouterr().out
    assert "before" in captured
    assert dataset.stacks.shape == (2, 6, 4)
    assert torch.allclose(dataset.stacks.norm(dim=-1), torch.ones(2, 6), atol=1e-5)
    assert torch.all(dataset.refined_fidelities >= dataset.initial_fidelities - 1e-6)


def test_hamiltonian_solution_dataset_can_keep_multiple_solutions_per_target():
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=21,
    )

    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=22,
        solutions_per_target=2,
    )

    assert dataset.stacks.shape == (4, 6, 4)
    assert len(dataset.targets) == 4
    assert len({target.name for target in dataset.targets}) == 2
    assert torch.allclose(dataset.stacks.norm(dim=-1), torch.ones(4, 6), atol=1e-5)


def test_hamiltonian_solution_dataset_supports_three_entanglers():
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=23,
    )

    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_entanglers=3,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=24,
    )

    assert dataset.stacks.shape == (2, 8, 4)
    assert len(dataset.refinements[0].slot_labels) == 8
    assert torch.allclose(dataset.stacks.norm(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_hamiltonian_template_comparison_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=25,
    )

    result = run_hamiltonian_template_comparison(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=26,
    )
    print_hamiltonian_template_comparison(result)

    captured = capsys.readouterr().out
    assert isinstance(result, HamiltonianTemplateComparisonResult)
    assert "3 CZ" in captured
    assert result.rows[0].n_slots == 6
    assert result.rows[1].n_slots == 8


def test_hamiltonian_conditioned_circuit_diffusion_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=31,
    )
    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=32,
    )
    config = CircuitExperimentConfig(
        name="test-hamiltonian-conditioned",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )

    model, losses = train_hamiltonian_conditioned_circuit_diffusion(
        dataset,
        train_config=config.train,
        schedule=config.schedule,
        device="cpu",
        show_progress=False,
    )
    samples = sample_hamiltonian_conditioned_circuit_reverse(
        model,
        config.schedule,
        targets,
        n_samples_per_target=2,
        eta=0.0,
        device="cpu",
        max_batch_size=1,
    )
    assert len(losses) == 2
    assert samples.shape == (2, 2, 6, 4)
    assert torch.isfinite(samples).all()
    assert torch.allclose(samples.norm(dim=-1), torch.ones(2, 2, 6), atol=1e-5)

    baseline = run_hamiltonian_suite_benchmark(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        keep_fidelities=False,
        seed=33,
    )
    result = run_hamiltonian_conditioned_diffusion_benchmark(
        dataset,
        eval_targets=targets,
        config=config,
        device="cpu",
        show_progress=False,
        top_k=1,
    )
    assert isinstance(result, HamiltonianConditionedDiffusionResult)
    assert result.generated_by_target.shape == (2, 3, 6, 4)
    assert len(result.reports) == 2
    rows = summarize_hamiltonian_conditioned_diffusion(baseline, result)
    assert rows[-1].mode == "Hamiltonian-conditioned diffusion"
    print_hamiltonian_conditioned_diffusion(result)
    print_hamiltonian_conditioned_diffusion_summary(baseline, result)
    captured = capsys.readouterr().out
    assert "conditioned" in captured


def test_hamiltonian_conditioned_overfit_diagnostic_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=41,
    )
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=42,
    )
    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=43,
        solutions_per_target=2,
    )
    config = CircuitExperimentConfig(
        name="test-hamiltonian-overfit",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )
    result = run_hamiltonian_conditioned_overfit_diagnostic(
        dataset,
        heldout_targets=heldout_targets,
        config=config,
        device="cpu",
        show_progress=False,
        top_k=1,
    )

    assert isinstance(result, HamiltonianConditionedOverfitDiagnosticResult)
    assert len(result.losses) == 2
    assert len(result.train_targets) == 2
    assert result.train_generated_by_target.shape == (2, 3, 6, 4)
    assert result.heldout_generated_by_target.shape == (2, 3, 6, 4)
    rows = summarize_hamiltonian_conditioned_overfit_diagnostic(result)
    assert [row.mode for row in rows] == ["train targets", "heldout targets"]
    print_hamiltonian_conditioned_overfit_diagnostic(result)
    print_hamiltonian_conditioned_overfit_summary(result)
    captured = capsys.readouterr().out
    assert "heldout" in captured


def test_hamiltonian_conditioned_denoise_diagnostic_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=51,
    )
    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=52,
        solutions_per_target=2,
    )
    config = CircuitExperimentConfig(
        name="test-hamiltonian-denoise",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )
    result = run_hamiltonian_conditioned_denoise_diagnostic(
        dataset,
        config=config,
        device="cpu",
        show_progress=False,
        timesteps=(1, 2, 4),
        seed=53,
    )
    rows = summarize_hamiltonian_denoise_diagnostic(result)
    direct_rows = evaluate_hamiltonian_conditioned_denoising(
        result.model,
        dataset,
        config.schedule,
        timesteps=(1,),
        n_terms=4,
        device="cpu",
        seed=54,
    )
    reused = hamiltonian_denoise_diagnostic_from_model(
        result.model,
        dataset,
        config=config,
        losses=result.losses,
        device="cpu",
        timesteps=(1,),
        seed=55,
    )
    print_hamiltonian_denoise_diagnostic(result)

    captured = capsys.readouterr().out
    assert isinstance(result, HamiltonianDenoiseDiagnosticResult)
    assert "rel mse" in captured
    assert len(result.losses) == 2
    assert len(rows) == 3
    assert len(direct_rows) == 1
    assert len(reused.rows) == 1
    assert reused.losses == result.losses
    for row in [*rows, *direct_rows]:
        assert 1 <= row.timestep <= config.schedule.T
        assert row.mse >= 0.0
        assert row.zero_mse >= 0.0
        assert row.relative_mse >= 0.0
        assert -1.0001 <= row.cosine <= 1.0001
        assert row.target_norm >= 0.0
        assert row.pred_norm >= 0.0
        assert torch.isfinite(torch.tensor([row.mse, row.relative_mse, row.cosine])).all()


def test_hamiltonian_denoise_ablation_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=56,
    )
    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=57,
        solutions_per_target=2,
    )
    base = CircuitExperimentConfig(
        name="test-ablation",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )
    configs = [
        base,
        CircuitExperimentConfig(
            name="test-ablation-wider",
            schedule=base.schedule,
            train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=24, n_terms=4),
            sample_count=base.sample_count,
            eta=base.eta,
        ),
    ]
    result = run_hamiltonian_denoise_ablation(
        dataset,
        base_config=base,
        configs=configs,
        device="cpu",
        show_progress=False,
        timesteps=(1, 4),
        seed=58,
    )
    rows = summarize_hamiltonian_denoise_ablation(result)
    print_hamiltonian_denoise_ablation(result)

    captured = capsys.readouterr().out
    assert isinstance(result, HamiltonianDenoiseAblationResult)
    assert "pred/target" in captured
    assert len(result.diagnostics) == 2
    assert len(rows) == 2
    assert [row.name for row in rows] == ["test-ablation", "test-ablation-wider"]
    for row in rows:
        assert row.num_steps == 2
        assert row.hidden in {16, 24}
        assert row.final_loss >= 0.0
        assert row.t1_relative_mse >= 0.0
        assert row.final_relative_mse >= 0.0
        assert -1.0001 <= row.final_cosine <= 1.0001
        assert row.final_pred_target_norm_ratio >= 0.0


def test_hamiltonian_denoise_normalization_comparison_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=59,
    )
    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=60,
        solutions_per_target=2,
    )
    config = CircuitExperimentConfig(
        name="test-normalization",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )
    scale = estimate_hamiltonian_denoise_target_scale(
        dataset,
        config.schedule,
        batch_size=4,
        n_batches=2,
        n_terms=4,
        device="cpu",
        seed=61,
    )
    result = run_hamiltonian_denoise_normalization_comparison(
        dataset,
        base_config=config,
        device="cpu",
        show_progress=False,
        timesteps=(1, 4),
        seed=62,
    )
    rows = summarize_hamiltonian_denoise_normalization(result)
    print_hamiltonian_denoise_normalization(result)

    captured = capsys.readouterr().out
    assert isinstance(result, HamiltonianDenoiseNormalizationResult)
    assert "target scale" in captured
    assert scale > 0.0
    assert len(result.diagnostics) == 3
    assert [row.variant for row in rows] == ["unnormalized", "normalized", "normalized+wider"]
    assert rows[0].target_scale == 1.0
    assert rows[1].target_scale > 0.0
    assert rows[2].target_scale == rows[1].target_scale
    for row in rows:
        assert row.final_loss >= 0.0
        assert row.final_relative_mse >= 0.0
        assert -1.0001 <= row.final_cosine <= 1.0001
        assert row.final_pred_target_norm_ratio >= 0.0


def test_hamiltonian_skeleton_denoise_comparison_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=63,
    )
    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=64,
        solutions_per_target=2,
    )
    config = CircuitExperimentConfig(
        name="test-skeleton-denoise",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )
    model, losses = train_hamiltonian_skeleton_conditioned_circuit_diffusion(
        dataset,
        labels,
        train_config=config.train,
        schedule=config.schedule,
        device="cpu",
        show_progress=False,
    )
    direct_rows = evaluate_hamiltonian_skeleton_conditioned_denoising(
        model,
        dataset,
        labels,
        config.schedule,
        timesteps=(1,),
        n_terms=4,
        device="cpu",
        seed=65,
    )
    diagnostic = run_hamiltonian_skeleton_denoise_diagnostic(
        dataset,
        labels,
        config=config,
        device="cpu",
        show_progress=False,
        timesteps=(1, 4),
        seed=66,
    )
    comparison = run_hamiltonian_skeleton_denoise_comparison(
        dataset,
        labels,
        base_config=config,
        device="cpu",
        show_progress=False,
        timesteps=(1, 4),
        seed=67,
    )
    rows = summarize_hamiltonian_skeleton_denoise_comparison(comparison)
    print_hamiltonian_skeleton_denoise_comparison(comparison)

    captured = capsys.readouterr().out
    assert "H+slot labels" in captured
    assert isinstance(comparison, HamiltonianSkeletonDenoiseComparisonResult)
    assert len(losses) == 2
    assert len(direct_rows) == 1
    assert len(diagnostic.rows) == 2
    assert len(rows) == 2
    assert [row.variant for row in rows] == ["H-only", "H+slot labels"]
    for row in rows:
        assert row.final_loss >= 0.0
        assert row.final_relative_mse >= 0.0
        assert -1.0001 <= row.final_cosine <= 1.0001
        assert row.final_pred_target_norm_ratio >= 0.0


def test_hamiltonian_slotwise_denoise_comparison_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=68,
    )
    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=69,
        solutions_per_target=2,
    )
    config = CircuitExperimentConfig(
        name="test-slotwise-denoise",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )
    model, losses = train_hamiltonian_slotwise_circuit_diffusion(
        dataset,
        train_config=config.train,
        schedule=config.schedule,
        device="cpu",
        show_progress=False,
    )
    diagnostic = run_hamiltonian_slotwise_denoise_diagnostic(
        dataset,
        config=config,
        device="cpu",
        show_progress=False,
        timesteps=(1, 4),
        seed=70,
    )
    comparison = run_hamiltonian_slotwise_denoise_comparison(
        dataset,
        base_config=config,
        device="cpu",
        show_progress=False,
        timesteps=(1, 4),
        seed=71,
    )
    rows = summarize_hamiltonian_slotwise_denoise_comparison(comparison)
    print_hamiltonian_slotwise_denoise_comparison(comparison)

    captured = capsys.readouterr().out
    assert "slot-wise MLP" in captured
    assert isinstance(comparison, HamiltonianSlotwiseDenoiseComparisonResult)
    assert len(losses) == 2
    assert model.n_slots == 6
    assert len(diagnostic.rows) == 2
    assert len(rows) == 2
    assert [row.variant for row in rows] == ["flat MLP", "slot-wise MLP"]
    for row in rows:
        assert row.final_loss >= 0.0
        assert row.final_relative_mse >= 0.0
        assert -1.0001 <= row.final_cosine <= 1.0001
        assert row.final_pred_target_norm_ratio >= 0.0


def test_hamiltonian_token_denoiser_and_proposal_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=72,
    )
    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=73,
        solutions_per_target=2,
    )
    config = CircuitExperimentConfig(
        name="test-token-denoise",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )
    model, losses = train_hamiltonian_token_circuit_diffusion(
        dataset,
        train_config=config.train,
        schedule=config.schedule,
        device="cpu",
        show_progress=False,
    )
    diagnostic = run_hamiltonian_token_denoise_diagnostic(
        dataset,
        config=config,
        device="cpu",
        show_progress=False,
        timesteps=(1, 4),
        seed=74,
    )
    comparison = run_hamiltonian_token_denoise_comparison(
        dataset,
        base_config=config,
        device="cpu",
        show_progress=False,
        timesteps=(1, 4),
        seed=75,
    )
    suite = run_hamiltonian_suite_benchmark(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        keep_fidelities=False,
        seed=76,
    )
    proposal = run_hamiltonian_token_conditioned_diffusion_benchmark(
        dataset,
        eval_targets=targets,
        config=config,
        device="cpu",
        show_progress=False,
        top_k=1,
    )
    rows = summarize_hamiltonian_token_denoise_comparison(comparison)
    summary = summarize_hamiltonian_token_conditioned_diffusion(suite, proposal)
    print_hamiltonian_token_denoise_comparison(comparison)
    print_hamiltonian_conditioned_diffusion(proposal)
    print_hamiltonian_token_conditioned_diffusion_summary(suite, proposal)

    captured = capsys.readouterr().out
    assert "circuit-token" in captured
    assert isinstance(comparison, HamiltonianTokenDenoiseComparisonResult)
    assert isinstance(proposal, HamiltonianConditionedDiffusionResult)
    assert len(losses) == 2
    assert model.n_slots == 6
    assert len(diagnostic.rows) == 2
    assert len(rows) == 2
    assert [row.variant for row in rows] == ["flat MLP", "circuit-token"]
    assert summary[-1].mode == "Hamiltonian circuit-token diffusion"
    assert proposal.generated_by_target.shape == (2, 3, 6, 4)
    assert torch.isfinite(proposal.generated_by_target).all()
    assert torch.allclose(proposal.generated_by_target.norm(dim=-1), torch.ones(2, 3, 6), atol=1e-5)
    for row in rows:
        assert row.final_loss >= 0.0
        assert row.final_relative_mse >= 0.0
        assert -1.0001 <= row.final_cosine <= 1.0001
        assert row.final_pred_target_norm_ratio >= 0.0


def test_hamiltonian_token_heldout_benchmark_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    train_targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=77,
    )
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=78,
    )
    dataset = generate_hamiltonian_solution_dataset(
        train_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=79,
        solutions_per_target=2,
    )
    config = CircuitExperimentConfig(
        name="test-token-heldout",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )
    heldout_suite = run_hamiltonian_suite_benchmark(
        heldout_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        keep_fidelities=False,
        seed=80,
    )
    result = run_hamiltonian_token_conditioned_overfit_diagnostic(
        dataset,
        heldout_targets=heldout_targets,
        config=config,
        device="cpu",
        show_progress=False,
        top_k=1,
    )
    rows = summarize_hamiltonian_conditioned_overfit_diagnostic(result)
    comparison = summarize_hamiltonian_token_heldout_comparison(heldout_suite, result)
    print_hamiltonian_conditioned_overfit_diagnostic(result)
    print_hamiltonian_conditioned_overfit_summary(result)
    print_hamiltonian_token_heldout_comparison_summary(heldout_suite, result)

    captured = capsys.readouterr().out
    assert "heldout targets" in captured
    assert "Hamiltonian circuit-token heldout" in captured
    assert isinstance(result, HamiltonianConditionedOverfitDiagnosticResult)
    assert len(rows) == 2
    assert len(comparison) == 5
    assert comparison[-1].mode == "Hamiltonian circuit-token heldout"
    assert result.train_generated_by_target.shape == (2, 3, 6, 4)
    assert result.heldout_generated_by_target.shape == (2, 3, 6, 4)
    assert torch.isfinite(result.heldout_generated_by_target).all()
    assert torch.allclose(result.heldout_generated_by_target.norm(dim=-1), torch.ones(2, 3, 6), atol=1e-5)


def test_hamiltonian_token_data_scale_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=81,
    )
    config = CircuitExperimentConfig(
        name="test-token-scale",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )

    result = run_hamiltonian_token_data_scale_benchmark(
        train_target_counts=(1, 2),
        heldout_targets=heldout_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        config=config,
        coefficient_scale=0.15,
        time=0.4,
        train_seed=82,
        dataset_seed=83,
        heldout_baseline_seed=84,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        solutions_per_target=2,
        device="cpu",
        show_progress=False,
    )
    rows = summarize_hamiltonian_token_data_scale(result)
    print_hamiltonian_token_data_scale_summary(result)

    captured = capsys.readouterr().out
    assert "n train" in captured
    assert isinstance(result, HamiltonianTokenDataScaleResult)
    assert len(rows) == 2
    assert [row.n_train_targets for row in rows] == [1, 2]
    assert set(result.diagnostics) == {1, 2}
    assert len(result.heldout_baseline.benchmarks) == 2
    for row in rows:
        assert row.n_solution_stacks >= row.n_train_targets
        assert row.final_loss >= 0.0
        assert 0.0 <= row.heldout_mean_best <= 1.0001
        assert 0.0 <= row.heldout_success_95 <= 1.0


def test_hamiltonian_token_stack_data_scale_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=1,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=841,
    )
    config = CircuitExperimentConfig(
        name="test-token-stack-scale",
        schedule=DiffusionSchedule(T=3, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=2, num_steps=1, hidden=16, n_terms=4),
        sample_count=2,
        eta=0.0,
    )

    result = run_hamiltonian_token_stack_data_scale_benchmark(
        settings=((1, 1), (2, 1)),
        heldout_targets=heldout_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        config=config,
        coefficient_scale=0.15,
        time=0.4,
        train_seed=842,
        dataset_seed=843,
        heldout_baseline_seed=844,
        n_entanglers=3,
        n_random_candidates=8,
        n_analytic_gates=4,
        n_haar_gates=4,
        top_k=1,
        refinement_steps=1,
        refinement_lr=0.02,
        solution_selection="min_local_rotation",
        selection_pool_size=2,
        train_steps=1,
        device="cpu",
        show_progress=False,
    )
    rows = summarize_hamiltonian_token_stack_data_scale(result)
    print_hamiltonian_token_stack_data_scale_summary(result)

    captured = capsys.readouterr().out
    assert "sol/target" in captured
    assert isinstance(result, HamiltonianTokenStackDataScaleResult)
    assert len(rows) == 2
    assert set(result.diagnostics) == {(1, 1), (2, 1)}
    assert len(result.heldout_baseline.benchmarks) == 1
    assert [row.n_slots for row in rows] == [8, 8]
    assert all(next(diagnostic.model.parameters()).device.type == "cpu" for diagnostic in result.diagnostics.values())
    for row in rows:
        assert row.n_entanglers == 3
        assert row.solutions_per_target == 1
        assert row.n_solution_stacks == row.n_train_targets
        assert row.final_loss >= 0.0
        assert 0.0 <= row.heldout_mean_best <= 1.0001


def test_hamiltonian_token_stack_training_budget_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=1,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=851,
    )
    config = CircuitExperimentConfig(
        name="test-token-stack-budget",
        schedule=DiffusionSchedule(T=3, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=2, num_steps=1, hidden=16, n_terms=4),
        sample_count=2,
        eta=0.0,
    )

    result = run_hamiltonian_token_stack_training_budget_benchmark(
        train_target_counts=(1, 2),
        train_step_counts=(1, 2),
        heldout_targets=heldout_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        config=config,
        coefficient_scale=0.15,
        time=0.4,
        train_seed=852,
        dataset_seed=853,
        heldout_baseline_seed=854,
        n_entanglers=3,
        n_random_candidates=8,
        n_analytic_gates=4,
        n_haar_gates=4,
        top_k=1,
        refinement_steps=1,
        refinement_lr=0.02,
        solution_selection="min_local_rotation",
        selection_pool_size=2,
        device="cpu",
        show_progress=False,
    )
    rows = summarize_hamiltonian_token_stack_training_budget(result)
    print_hamiltonian_token_stack_training_budget_summary(result)

    captured = capsys.readouterr().out
    assert "steps" in captured
    assert isinstance(result, HamiltonianTokenStackTrainingBudgetResult)
    assert len(rows) == 4
    assert set(result.diagnostics) == {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert len(result.heldout_baseline.benchmarks) == 1
    assert result.train_dataset.stacks.shape[1] == 8
    assert all(next(diagnostic.model.parameters()).device.type == "cpu" for diagnostic in result.diagnostics.values())
    for row in rows:
        assert row.n_train_targets in {1, 2}
        assert row.num_steps in {1, 2}
        assert row.n_solution_stacks == row.n_train_targets
        assert row.final_loss >= 0.0
        assert 0.0 <= row.heldout_mean_best <= 1.0001


def test_hamiltonian_token_training_budget_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=85,
    )
    config = CircuitExperimentConfig(
        name="test-token-budget",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )

    result = run_hamiltonian_token_training_budget_benchmark(
        train_target_count=2,
        train_step_counts=(2, 3),
        heldout_targets=heldout_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        config=config,
        coefficient_scale=0.15,
        time=0.4,
        train_seed=86,
        dataset_seed=87,
        heldout_baseline_seed=88,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        solutions_per_target=2,
        device="cpu",
        show_progress=False,
    )
    rows = summarize_hamiltonian_token_training_budget(result)
    print_hamiltonian_token_training_budget_summary(result)

    captured = capsys.readouterr().out
    assert "steps" in captured
    assert isinstance(result, HamiltonianTokenTrainingBudgetResult)
    assert len(rows) == 2
    assert [row.num_steps for row in rows] == [2, 3]
    assert set(result.diagnostics) == {2, 3}
    assert result.train_dataset.stacks.shape == (4, 6, 4)
    for row in rows:
        assert row.n_train_targets == 2
        assert row.n_solution_stacks == 4
        assert row.final_loss >= 0.0
        assert 0.0 <= row.heldout_mean_best <= 1.0001
        assert 0.0 <= row.heldout_success_95 <= 1.0


def test_hamiltonian_token_template_comparison_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=1,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=891,
    )
    config = CircuitExperimentConfig(
        name="test-token-template",
        schedule=DiffusionSchedule(T=3, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=2, num_steps=1, hidden=16, n_terms=4),
        sample_count=2,
        eta=0.0,
    )

    result = run_hamiltonian_token_template_comparison(
        train_target_count=1,
        train_steps=1,
        heldout_targets=heldout_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        config=config,
        coefficient_scale=0.15,
        time=0.4,
        train_seed=892,
        two_dataset_seed=893,
        three_dataset_seed=894,
        two_baseline_seed=895,
        three_baseline_seed=896,
        n_random_candidates=8,
        n_analytic_gates=4,
        n_haar_gates=4,
        top_k=1,
        refinement_steps=1,
        refinement_lr=0.02,
        solutions_per_target=1,
        device="cpu",
        show_progress=False,
    )
    rows = summarize_hamiltonian_token_template_comparison(result)
    print_hamiltonian_token_template_comparison(result)

    captured = capsys.readouterr().out
    assert "3 CZ" in captured
    assert isinstance(result, HamiltonianTokenTemplateComparisonResult)
    assert len(rows) == 2
    assert [row.n_slots for row in rows] == [6, 8]
    assert result.two_entangler.train_dataset.stacks.shape == (1, 6, 4)
    assert result.three_entangler.train_dataset.stacks.shape == (1, 8, 4)
    for row in rows:
        assert row.n_train_targets == 1
        assert row.n_heldout_targets == 1
        assert row.n_solution_stacks == 1
        assert row.final_loss >= 0.0
        assert 0.0 <= row.heldout_mean_best <= 1.0001


def test_hamiltonian_token_repeatability_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    config = CircuitExperimentConfig(
        name="test-token-repeat",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )

    result = run_hamiltonian_token_repeatability_benchmark(
        n_runs=2,
        train_target_count=2,
        heldout_target_count=2,
        train_steps=2,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        config=config,
        coefficient_scale=0.15,
        time=0.4,
        seed=89,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        solutions_per_target=2,
        device="cpu",
        show_progress=False,
    )
    rows = summarize_hamiltonian_token_repeatability(result)
    print_hamiltonian_token_repeatability_summary(result)

    captured = capsys.readouterr().out
    assert "token-gen" in captured
    assert isinstance(result, HamiltonianTokenRepeatabilityResult)
    assert len(rows) == 2
    assert len(result.budget_results) == 2
    assert [row.run for row in rows] == [0, 1]
    for row in rows:
        assert row.num_steps == 2
        assert row.n_train_targets == 2
        assert row.n_heldout_targets == 2
        assert row.n_solution_stacks == 4
        assert row.final_loss >= 0.0
        assert 0.0 <= row.heldout_mean_best <= 1.0001
        assert 0.0 <= row.heldout_success_95 <= 1.0


def test_hamiltonian_repeatability_refinement_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    config = CircuitExperimentConfig(
        name="test-repeat-refine",
        schedule=DiffusionSchedule(T=4, beta_start=1e-4, beta_end=0.005, kind="linear"),
        train=CircuitTrainConfig(batch_size=4, num_steps=2, hidden=16, n_terms=4),
        sample_count=3,
        eta=0.2,
    )

    repeatability = run_hamiltonian_token_repeatability_benchmark(
        n_runs=1,
        train_target_count=2,
        heldout_target_count=2,
        train_steps=2,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        config=config,
        coefficient_scale=0.15,
        time=0.4,
        seed=90,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        solutions_per_target=2,
        device="cpu",
        show_progress=False,
    )
    result = run_hamiltonian_repeatability_refinement_benchmark(
        repeatability,
        generated_gates=centers,
        refinement_steps=2,
        refinement_lr=0.02,
        threshold=0.5,
    )
    rows = summarize_hamiltonian_repeatability_refinement(result)
    headline = summarize_hamiltonian_level1_headline(result)
    print_hamiltonian_repeatability_refinement(result)
    print_hamiltonian_repeatability_refinement_summary(result)
    print_hamiltonian_level1_headline_table(result)

    captured = capsys.readouterr().out
    assert "token" in captured
    assert "generated-search" in captured
    assert "move mean" in captured
    assert "proposal advantage" in captured
    assert isinstance(result, HamiltonianRepeatabilityRefinementResult)
    assert isinstance(headline, HamiltonianLevel1HeadlineResult)
    assert len(rows) == 4
    assert len(headline.rows) == 2
    assert headline.n_runs == 1
    assert headline.threshold == 0.5
    assert {row.source for row in rows} == {"token", "generated-search"}
    assert {row.source for row in headline.rows} == {"token", "generated-search"}
    for row in rows:
        assert row.run == 0
        assert row.target.startswith("pauli-")
        assert 0.0 <= row.initial_fidelity <= 1.0001
        assert 0.0 <= row.refined_fidelity <= 1.0001
        assert row.steps_to_threshold >= -1
        assert len(row.slot_movements) == 6
        assert row.movement_mean >= 0.0
        assert row.movement_max >= row.movement_mean
        assert all(value >= 0.0 for value in row.slot_movements)
    for row in headline.rows:
        assert row.n_targets == 2
        assert 0.0 <= row.proposal_mean <= 1.0001
        assert 0.0 <= row.refined_mean <= 1.0001
        assert 0.0 <= row.refinement_success <= 1.0
        assert row.mean_movement >= 0.0


def test_hamiltonian_stack_predictor_shapes_and_smoke_training(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=9,
    )
    dataset = generate_hamiltonian_solution_dataset(
        targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=10,
    )
    features = hamiltonian_target_features(dataset.targets)
    model = HamiltonianStackPredictor(input_dim=features.shape[1], hidden=16)
    predicted = model(features)

    assert predicted.shape == (2, 6, 4)
    assert torch.allclose(predicted.norm(dim=-1), torch.ones(2, 6), atol=1e-5)

    result = run_hamiltonian_supervised_baseline(
        dataset,
        config=HamiltonianSupervisedTrainConfig(hidden=16, num_steps=2, lr=1e-3),
        device="cpu",
        show_progress=False,
        refine=False,
    )
    print_hamiltonian_supervised_summary(result)
    evaluated = evaluate_hamiltonian_stack_predictor(result.model, dataset.targets, device="cpu")

    captured = capsys.readouterr().out
    assert "mean raw" in captured
    assert len(result.losses) == 2
    assert result.predicted_stacks.shape == (2, 6, 4)
    assert evaluated.raw_fidelities.shape == (2,)
    assert torch.isfinite(result.raw_fidelities).all()


def test_hamiltonian_supervised_split_baseline_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    train_targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=11,
    )
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=12,
    )
    train_dataset = generate_hamiltonian_solution_dataset(
        train_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=13,
    )

    result = run_hamiltonian_supervised_split_baseline(
        train_dataset,
        heldout_targets,
        config=HamiltonianSupervisedTrainConfig(hidden=16, num_steps=2, lr=1e-3),
        device="cpu",
        show_progress=False,
        refine=False,
    )
    print_hamiltonian_supervised_split_summary(result)

    captured = capsys.readouterr().out
    assert "heldout" in captured
    assert result.train.raw_fidelities.shape == (2,)
    assert result.heldout.raw_fidelities.shape == (2,)


def test_hamiltonian_seed_ablation_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    train_targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=21,
    )
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=22,
    )
    train_dataset = generate_hamiltonian_solution_dataset(
        train_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=23,
    )
    supervised = run_hamiltonian_supervised_split_baseline(
        train_dataset,
        heldout_targets,
        config=HamiltonianSupervisedTrainConfig(hidden=16, num_steps=2, lr=1e-3),
        device="cpu",
        show_progress=False,
        refine=False,
    )
    heldout_suite = run_hamiltonian_suite_benchmark(
        heldout_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        keep_fidelities=False,
        seed=24,
    )

    ablation = run_hamiltonian_seed_ablation(
        heldout_targets,
        supervised.heldout.predicted_stacks,
        heldout_suite,
        clifford_gates=centers,
        generated_gates=centers,
        refinement_steps=2,
        refinement_lr=0.02,
        threshold=0.5,
        seed=25,
    )
    print_hamiltonian_seed_ablation(ablation)
    print_hamiltonian_seed_ablation_summary(ablation)

    captured = capsys.readouterr().out
    assert "mlp" in captured
    assert "haar" in captured
    assert len(ablation.rows) == 8
    assert {row.seed_type for row in ablation.rows} == {
        "mlp",
        "generated-search",
        "clifford-search",
        "haar",
    }
    assert all(0.0 <= row.initial_fidelity <= 1.0 for row in ablation.rows)
    assert all(0.0 <= row.refined_fidelity <= 1.0 for row in ablation.rows)


def test_hamiltonian_prior_search_smoke(capsys):
    data_config = DataConfig(kind="clifford")
    centers = centers_for_config(data_config, device="cpu")
    labels = center_names_for_config(data_config)
    train_targets = make_random_pauli_hamiltonian_targets(
        n_targets=3,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=31,
    )
    heldout_targets = make_random_pauli_hamiltonian_targets(
        n_targets=2,
        terms=("XI", "IZ", "XX", "ZZ"),
        coefficient_scale=0.15,
        time=0.4,
        seed=32,
    )
    train_dataset = generate_hamiltonian_solution_dataset(
        train_targets,
        clifford_gates=centers,
        clifford_labels=labels,
        generated_gates=centers,
        generated_labels=labels,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        refinement_steps=2,
        refinement_lr=0.02,
        seed=33,
    )
    prior = train_hamiltonian_slot_prior(
        train_dataset,
        labels,
        config=HamiltonianPriorTrainConfig(hidden=16, num_steps=2, lr=1e-3),
        device="cpu",
        show_progress=False,
    )
    benchmark = run_hamiltonian_prior_search_benchmark(
        prior,
        heldout_targets,
        local_gates=centers,
        local_labels=labels,
        n_candidates=16,
        top_k=1,
        seed=34,
    )
    print_hamiltonian_prior_search(benchmark)
    print_hamiltonian_prior_search_summary(benchmark)
    mixture = run_hamiltonian_prior_mixture_sweep(
        prior,
        heldout_targets,
        local_gates=centers,
        local_labels=labels,
        alphas=(0.0, 1.0),
        n_candidates=16,
        top_k=1,
        seed=35,
    )
    print_hamiltonian_prior_mixture_summary(mixture)
    refined = refine_hamiltonian_prior_mixture(
        mixture,
        local_gates=centers,
        refinement_steps=2,
        refinement_lr=0.02,
        threshold=0.5,
    )
    print_hamiltonian_mixture_refinement_summary(refined)
    budgeted = refine_hamiltonian_prior_mixture_budget_sweep(
        mixture,
        local_gates=centers,
        budgets=(1, 2),
        refinement_lr=0.02,
        threshold=0.5,
    )
    print_hamiltonian_budget_refinement_summary(budgeted)

    captured = capsys.readouterr().out
    assert "learned prior" in captured
    assert "alpha" in captured
    assert len(prior.losses) == 2
    assert 0.0 <= prior.train_accuracy <= 1.0
    assert len(benchmark.benchmarks) == 2
    assert set(mixture.alpha_results) == {0.0, 1.0}
    assert len(refined.rows) == 4
    assert len(budgeted.rows) == 8
    assert all(item.prior_report.candidates for item in benchmark.benchmarks)
    assert all(0.0 <= item.prior_report.candidates[0].fidelity <= 1.0 for item in benchmark.benchmarks)
    assert all(0.0 <= row.refined_fidelity <= 1.0 for row in refined.rows)
    assert all(0.0 <= row.refined_fidelity <= 1.0 for row in budgeted.rows)
