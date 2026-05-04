import torch

from su2diffusion.circuit import (
    CircuitDenoiser,
    CircuitExperimentConfig,
    CircuitTrainConfig,
    TargetConditionedCircuitDenoiser,
    TargetLabelConditionedCircuitDenoiser,
    circuit_forward_heat_target,
    generate_solution_stack_dataset,
    get_circuit_experiment_config,
    print_joint_circuit_comparison_summary,
    print_solution_stack_circuit_comparison_summary,
    print_solution_stack_dataset_summary,
    print_skeleton_local_refinement_summary,
    print_target_conditioned_circuit_comparison_summary,
    print_target_conditioned_learning_curve,
    print_target_conditioned_overfit_summary,
    print_target_label_conditioned_circuit_comparison_summary,
    run_circuit_experiment,
    run_joint_circuit_proposal_benchmark,
    run_solution_stack_circuit_experiment,
    run_skeleton_local_refinement_benchmark,
    run_target_conditioned_circuit_proposal_benchmark,
    run_target_conditioned_learning_curve,
    run_target_conditioned_overfit_diagnostic,
    run_target_conditioned_solution_stack_circuit_experiment,
    run_target_conditioned_synthetic_circuit_experiment,
    run_target_label_conditioned_skeleton_benchmark,
    sample_circuit_reverse,
    sample_near_clifford_circuit_stacks,
    sample_target_conditioned_circuit_reverse,
    sample_target_label_conditioned_circuit_reverse,
    sample_target_label_conditioned_circuit_reverse_from_skeleton,
    synthesize_unitary_from_circuit_stack_report,
    target_unitary_features,
    train_circuit_heat_kernel_model_on_stacks,
    train_target_conditioned_circuit_heat_kernel_model,
    train_target_conditioned_circuit_heat_kernel_model_synthetic,
    train_target_label_conditioned_circuit_heat_kernel_model_synthetic,
)
from su2diffusion.data import DataConfig, center_names_for_config, centers_for_config
from su2diffusion.diffusion import DiffusionSchedule
from su2diffusion.synthesis import (
    make_near_clifford_two_entangler_circuit_targets,
    run_near_clifford_two_entangler_benchmark,
    two_qubit_gate,
)


def test_near_clifford_circuit_sampler_returns_unit_stacks():
    config = DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    centers = centers_for_config(config, device="cpu")
    names = center_names_for_config(config)

    q_stack, labels = sample_near_clifford_circuit_stacks(
        5,
        centers=centers,
        center_names=names,
        sigma=config.sigma_data,
    )

    assert q_stack.shape == (5, 6, 4)
    assert len(labels) == 5
    assert all(len(row) == 6 for row in labels)
    assert torch.allclose(q_stack.norm(dim=-1), torch.ones(5, 6), atol=1e-6)


def test_circuit_forward_heat_target_shapes_and_norms():
    config = DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    centers = centers_for_config(config, device="cpu")
    names = center_names_for_config(config)
    schedule = DiffusionSchedule(T=8)
    q0_stack, _ = sample_near_clifford_circuit_stacks(4, centers=centers, center_names=names)
    t_idx = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    qt_stack, eps_target = circuit_forward_heat_target(q0_stack, t_idx, schedule=schedule, n_terms=8)

    assert qt_stack.shape == (4, 6, 4)
    assert eps_target.shape == (4, 6, 3)
    assert torch.isfinite(eps_target).all()
    assert torch.allclose(qt_stack.norm(dim=-1), torch.ones(4, 6), atol=1e-5)


def test_circuit_denoiser_output_shape():
    model = CircuitDenoiser(T=8, hidden=16)
    q_stack = torch.randn(3, 6, 4)
    t_idx = torch.tensor([1, 2, 3], dtype=torch.long)

    eps = model(q_stack, t_idx)

    assert eps.shape == (3, 6, 3)


def test_target_conditioned_circuit_denoiser_output_shape():
    model = TargetConditionedCircuitDenoiser(T=8, hidden=16)
    q_stack = torch.randn(3, 6, 4)
    t_idx = torch.tensor([1, 2, 3], dtype=torch.long)
    features = torch.randn(3, 32)

    eps = model(q_stack, t_idx, features)

    assert eps.shape == (3, 6, 3)


def test_target_label_conditioned_circuit_denoiser_output_shape():
    model = TargetLabelConditionedCircuitDenoiser(T=8, hidden=16, num_labels=24)
    q_stack = torch.randn(3, 6, 4)
    t_idx = torch.tensor([1, 2, 3], dtype=torch.long)
    features = torch.randn(3, 32)
    labels = torch.randint(0, 24, (3, 6))

    eps = model(q_stack, t_idx, features, labels)

    assert eps.shape == (3, 6, 3)


def test_circuit_reverse_sampler_returns_normalized_stacks():
    schedule = DiffusionSchedule(T=4)
    model = CircuitDenoiser(T=schedule.T, hidden=16)

    q_stack = sample_circuit_reverse(model, schedule, n_samples=6, eta=0.0, device="cpu")

    assert q_stack.shape == (6, 6, 4)
    assert torch.isfinite(q_stack).all()
    assert torch.allclose(q_stack.norm(dim=-1), torch.ones(6, 6), atol=1e-5)


def test_target_conditioned_reverse_sampler_returns_normalized_stacks():
    schedule = DiffusionSchedule(T=4)
    model = TargetConditionedCircuitDenoiser(T=schedule.T, hidden=16)
    targets = torch.stack([torch.eye(4, dtype=torch.complex64), two_qubit_gate("cz", device="cpu")])

    q_stack = sample_target_conditioned_circuit_reverse(
        model,
        schedule,
        target_unitaries=targets,
        n_samples_per_target=3,
        eta=0.0,
        device="cpu",
    )

    assert q_stack.shape == (2, 3, 6, 4)
    assert torch.isfinite(q_stack).all()
    assert torch.allclose(q_stack.norm(dim=-1), torch.ones(2, 3, 6), atol=1e-5)


def test_target_label_conditioned_reverse_sampler_returns_normalized_stacks():
    schedule = DiffusionSchedule(T=4)
    model = TargetLabelConditionedCircuitDenoiser(T=schedule.T, hidden=16, num_labels=24)
    targets = torch.stack([torch.eye(4, dtype=torch.complex64), two_qubit_gate("cz", device="cpu")])
    labels = torch.randint(0, 24, (2, 6))

    q_stack = sample_target_label_conditioned_circuit_reverse(
        model,
        schedule,
        target_unitaries=targets,
        slot_labels=labels,
        n_samples_per_target=3,
        eta=0.0,
        device="cpu",
    )

    assert q_stack.shape == (2, 3, 6, 4)
    assert torch.isfinite(q_stack).all()
    assert torch.allclose(q_stack.norm(dim=-1), torch.ones(2, 3, 6), atol=1e-5)


def test_target_label_conditioned_skeleton_sampler_returns_normalized_stacks():
    schedule = DiffusionSchedule(T=4)
    model = TargetLabelConditionedCircuitDenoiser(T=schedule.T, hidden=16, num_labels=24)
    targets = torch.stack([torch.eye(4, dtype=torch.complex64), two_qubit_gate("cz", device="cpu")])
    labels = torch.randint(0, 24, (2, 6))
    centers = centers_for_config(DataConfig(kind="clifford"), device="cpu")

    q_stack = sample_target_label_conditioned_circuit_reverse_from_skeleton(
        model,
        schedule,
        target_unitaries=targets,
        slot_labels=labels,
        centers=centers,
        n_samples_per_target=3,
        eta=0.0,
        init_noise_scale=0.01,
        device="cpu",
    )

    assert q_stack.shape == (2, 3, 6, 4)
    assert torch.isfinite(q_stack).all()
    assert torch.allclose(q_stack.norm(dim=-1), torch.ones(2, 3, 6), atol=1e-5)


def test_circuit_experiment_smoke_runs_on_cpu():
    config = CircuitExperimentConfig(
        name="tiny-circuit",
        schedule=DiffusionSchedule(T=4),
        train=CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4),
        sample_count=8,
    )

    result = run_circuit_experiment(config, device="cpu", show_progress=False)

    assert result.generated_stochastic.shape == (8, 6, 4)
    assert len(result.losses) == 1


def test_circuit_config_registry_has_smoke_config():
    config = get_circuit_experiment_config("smoke-circuit-near-clifford")

    assert config.name == "smoke-circuit-near-clifford"
    assert config.sample_count > 0


def test_circuit_stack_report_and_joint_comparison_are_finite(capsys):
    data_config = DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    centers = centers_for_config(data_config, device="cpu")
    names = center_names_for_config(data_config)
    targets = make_near_clifford_two_entangler_circuit_targets(
        centers,
        names,
        n_targets=1,
        perturb_scale=0.05,
        seed=1,
    )
    q_stacks, _ = sample_near_clifford_circuit_stacks(16, centers=centers, center_names=names)

    report = synthesize_unitary_from_circuit_stack_report(
        q_stacks,
        target_unitary=targets[0].unitary,
        target_name=targets[0].name,
        top_k=3,
    )

    assert len(report.candidates) == 3
    assert all(0.0 <= candidate.fidelity <= 1.0 for candidate in report.candidates)

    near_benchmarks = run_near_clifford_two_entangler_benchmark(
        clifford_gates=centers,
        clifford_labels=names,
        generated_gates=centers,
        generated_labels=names,
        n_targets=1,
        perturb_scale=0.05,
        n_random_candidates=32,
        n_analytic_gates=16,
        n_haar_gates=16,
        top_k=1,
        seed=2,
        keep_fidelities=False,
    )
    joint_reports = run_joint_circuit_proposal_benchmark(
        near_benchmarks,
        circuit_stacks=q_stacks,
        top_k=1,
        keep_fidelities=False,
    )
    print_joint_circuit_comparison_summary(near_benchmarks, joint_reports)

    captured = capsys.readouterr().out
    assert "joint circuit diffusion" in captured
    assert len(joint_reports) == 1
    assert joint_reports[0].candidates[0].fidelity <= 1.0


def test_solution_stack_dataset_generation_returns_refined_stacks(capsys):
    data_config = DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    centers = centers_for_config(data_config, device="cpu")
    names = center_names_for_config(data_config)

    dataset = generate_solution_stack_dataset(
        centers,
        names,
        n_targets=2,
        perturb_scale=0.02,
        candidate_count=16,
        refinement_steps=5,
        fidelity_threshold=0.0,
        seed=4,
    )
    print_solution_stack_dataset_summary(dataset)

    captured = capsys.readouterr().out
    assert "n stacks" in captured
    assert dataset.stacks.shape[1:] == (6, 4)
    assert dataset.stacks.shape[0] == dataset.fidelities.shape[0]
    assert torch.allclose(dataset.stacks.norm(dim=-1), torch.ones_like(dataset.stacks[..., 0]), atol=1e-5)


def test_train_circuit_model_on_solution_stacks_smoke():
    q_stacks = q_normalized_random_stacks(8)
    schedule = DiffusionSchedule(T=4)
    config = CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4)

    model, losses = train_circuit_heat_kernel_model_on_stacks(
        q_stacks,
        train_config=config,
        schedule=schedule,
        device="cpu",
        show_progress=False,
    )

    assert isinstance(model, CircuitDenoiser)
    assert len(losses) == 1


def test_solution_stack_circuit_experiment_and_summary(capsys):
    data_config = DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    centers = centers_for_config(data_config, device="cpu")
    names = center_names_for_config(data_config)
    q_stacks = q_normalized_random_stacks(8)
    config = CircuitExperimentConfig(
        name="tiny-solution-circuit",
        schedule=DiffusionSchedule(T=4),
        train=CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4),
        sample_count=8,
    )

    result = run_solution_stack_circuit_experiment(q_stacks, config, device="cpu", show_progress=False)
    near_benchmarks = run_near_clifford_two_entangler_benchmark(
        clifford_gates=centers,
        clifford_labels=names,
        generated_gates=centers,
        generated_labels=names,
        n_targets=1,
        perturb_scale=0.05,
        n_random_candidates=32,
        n_analytic_gates=16,
        n_haar_gates=16,
        top_k=1,
        seed=2,
        keep_fidelities=False,
    )
    random_reports = run_joint_circuit_proposal_benchmark(near_benchmarks, q_stacks, top_k=1, keep_fidelities=False)
    solution_reports = run_joint_circuit_proposal_benchmark(
        near_benchmarks,
        result.generated_stochastic,
        top_k=1,
        keep_fidelities=False,
    )
    print_solution_stack_circuit_comparison_summary(near_benchmarks, random_reports, solution_reports)

    captured = capsys.readouterr().out
    assert "solution-stack diffusion" in captured
    assert result.generated_stochastic.shape == (8, 6, 4)


def test_target_unitary_features_shape_and_dtype():
    targets = torch.stack([torch.eye(4, dtype=torch.complex64), two_qubit_gate("cz", device="cpu")])

    features = target_unitary_features(targets)

    assert features.shape == (2, 32)
    assert features.dtype == torch.float32


def test_train_target_conditioned_circuit_model_smoke():
    q_stacks = q_normalized_random_stacks(8)
    targets = torch.stack([torch.eye(4, dtype=torch.complex64)] * 8)
    schedule = DiffusionSchedule(T=4)
    config = CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4)

    model, losses = train_target_conditioned_circuit_heat_kernel_model(
        q_stacks,
        targets,
        train_config=config,
        schedule=schedule,
        device="cpu",
        show_progress=False,
    )

    assert isinstance(model, TargetConditionedCircuitDenoiser)
    assert len(losses) == 1


def test_train_synthetic_target_conditioned_circuit_model_smoke():
    schedule = DiffusionSchedule(T=4)
    config = CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4)

    model, losses = train_target_conditioned_circuit_heat_kernel_model_synthetic(
        train_config=config,
        schedule=schedule,
        device="cpu",
        show_progress=False,
    )

    assert isinstance(model, TargetConditionedCircuitDenoiser)
    assert len(losses) == 1


def test_train_synthetic_target_label_conditioned_circuit_model_smoke():
    schedule = DiffusionSchedule(T=4)
    config = CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4)

    model, losses = train_target_label_conditioned_circuit_heat_kernel_model_synthetic(
        train_config=config,
        schedule=schedule,
        device="cpu",
        show_progress=False,
    )

    assert isinstance(model, TargetLabelConditionedCircuitDenoiser)
    assert len(losses) == 1


def test_target_conditioned_experiment_and_summary(capsys):
    data_config = DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    centers = centers_for_config(data_config, device="cpu")
    names = center_names_for_config(data_config)
    dataset = generate_solution_stack_dataset(
        centers,
        names,
        n_targets=2,
        perturb_scale=0.02,
        candidate_count=16,
        refinement_steps=5,
        fidelity_threshold=0.0,
        seed=5,
    )
    near_benchmarks = run_near_clifford_two_entangler_benchmark(
        clifford_gates=centers,
        clifford_labels=names,
        generated_gates=centers,
        generated_labels=names,
        n_targets=1,
        perturb_scale=0.05,
        n_random_candidates=32,
        n_analytic_gates=16,
        n_haar_gates=16,
        top_k=1,
        seed=6,
        keep_fidelities=False,
    )
    config = CircuitExperimentConfig(
        name="tiny-target-conditioned-circuit",
        schedule=DiffusionSchedule(T=4),
        train=CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4),
        sample_count=4,
    )
    eval_targets = torch.stack([item.target.unitary for item in near_benchmarks])

    result = run_target_conditioned_solution_stack_circuit_experiment(
        dataset,
        eval_targets,
        config,
        device="cpu",
        show_progress=False,
    )
    random_reports = run_joint_circuit_proposal_benchmark(
        near_benchmarks,
        q_normalized_random_stacks(8),
        top_k=1,
        keep_fidelities=False,
    )
    solution_reports = run_joint_circuit_proposal_benchmark(
        near_benchmarks,
        dataset.stacks,
        top_k=1,
        keep_fidelities=False,
    )
    conditioned_reports = run_target_conditioned_circuit_proposal_benchmark(
        near_benchmarks,
        result.generated_stochastic_by_target,
        top_k=1,
        keep_fidelities=False,
    )
    print_target_conditioned_circuit_comparison_summary(
        near_benchmarks,
        random_reports,
        conditioned_reports,
        solution_joint_reports=solution_reports,
    )

    captured = capsys.readouterr().out
    assert "target-conditioned diffusion" in captured
    assert result.generated_stochastic_by_target.shape == (1, 4, 6, 4)
    assert len(conditioned_reports) == 1


def test_synthetic_target_conditioned_experiment_and_summary(capsys):
    data_config = DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    centers = centers_for_config(data_config, device="cpu")
    names = center_names_for_config(data_config)
    near_benchmarks = run_near_clifford_two_entangler_benchmark(
        clifford_gates=centers,
        clifford_labels=names,
        generated_gates=centers,
        generated_labels=names,
        n_targets=1,
        perturb_scale=0.05,
        n_random_candidates=32,
        n_analytic_gates=16,
        n_haar_gates=16,
        top_k=1,
        seed=7,
        keep_fidelities=False,
    )
    config = CircuitExperimentConfig(
        name="tiny-target-conditioned-synthetic-circuit",
        schedule=DiffusionSchedule(T=4),
        train=CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4),
        sample_count=4,
    )
    eval_targets = torch.stack([item.target.unitary for item in near_benchmarks])

    result = run_target_conditioned_synthetic_circuit_experiment(
        eval_targets,
        config,
        device="cpu",
        show_progress=False,
    )
    random_reports = run_joint_circuit_proposal_benchmark(
        near_benchmarks,
        q_normalized_random_stacks(8),
        top_k=1,
        keep_fidelities=False,
    )
    conditioned_reports = run_target_conditioned_circuit_proposal_benchmark(
        near_benchmarks,
        result.generated_stochastic_by_target,
        top_k=1,
        keep_fidelities=False,
    )
    print_target_conditioned_circuit_comparison_summary(
        near_benchmarks,
        random_reports,
        conditioned_reports,
    )

    captured = capsys.readouterr().out
    assert "target-conditioned diffusion" in captured
    assert result.generated_stochastic_by_target.shape == (1, 4, 6, 4)


def test_target_conditioned_overfit_diagnostic_smoke(capsys):
    config = CircuitExperimentConfig(
        name="tiny-target-conditioned-overfit",
        schedule=DiffusionSchedule(T=4),
        train=CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4),
        sample_count=4,
    )

    result = run_target_conditioned_overfit_diagnostic(
        config,
        n_targets=2,
        perturb_scale=0.02,
        device="cpu",
        show_progress=False,
        top_k=1,
    )
    print_target_conditioned_overfit_summary(result)

    captured = capsys.readouterr().out
    assert "target-conditioned overfit" in captured
    assert result.solution_stacks.shape == (2, 6, 4)
    assert result.generated_stochastic_by_target.shape == (2, 4, 6, 4)
    assert len(result.reports) == 2


def test_target_conditioned_learning_curve_smoke(capsys):
    config = CircuitExperimentConfig(
        name="tiny-target-conditioned-learning-curve",
        schedule=DiffusionSchedule(T=4),
        train=CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4),
        sample_count=4,
    )

    result = run_target_conditioned_learning_curve(
        config,
        target_counts=(1, 2),
        n_heldout_targets=2,
        perturb_scale=0.02,
        device="cpu",
        show_progress=False,
        top_k=1,
    )
    print_target_conditioned_learning_curve(result)

    captured = capsys.readouterr().out
    assert "n train" in captured
    assert [row.n_train_targets for row in result.rows] == [1, 1, 2, 2]
    assert {row.split for row in result.rows} == {"train", "heldout"}


def test_target_label_conditioned_skeleton_benchmark_smoke(capsys):
    data_config = DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    centers = centers_for_config(data_config, device="cpu")
    names = center_names_for_config(data_config)
    near_benchmarks = run_near_clifford_two_entangler_benchmark(
        clifford_gates=centers,
        clifford_labels=names,
        generated_gates=centers,
        generated_labels=names,
        n_targets=1,
        perturb_scale=0.05,
        n_random_candidates=32,
        n_analytic_gates=16,
        n_haar_gates=16,
        top_k=1,
        seed=8,
        keep_fidelities=False,
    )
    config = CircuitExperimentConfig(
        name="tiny-target-label-conditioned",
        schedule=DiffusionSchedule(T=4),
        train=CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4),
        sample_count=4,
    )
    target_only = run_target_conditioned_synthetic_circuit_experiment(
        torch.stack([item.target.unitary for item in near_benchmarks]),
        config,
        device="cpu",
        show_progress=False,
    )
    target_reports = run_target_conditioned_circuit_proposal_benchmark(
        near_benchmarks,
        target_only.generated_stochastic_by_target,
        top_k=1,
        keep_fidelities=False,
    )
    label_result = run_target_label_conditioned_skeleton_benchmark(
        near_benchmarks,
        config,
        device="cpu",
        show_progress=False,
        top_k=1,
    )
    random_reports = run_joint_circuit_proposal_benchmark(
        near_benchmarks,
        q_normalized_random_stacks(8),
        top_k=1,
        keep_fidelities=False,
    )
    print_target_label_conditioned_circuit_comparison_summary(
        near_benchmarks,
        random_reports,
        target_reports,
        label_result.reports,
    )

    captured = capsys.readouterr().out
    assert "target+labels diffusion" in captured
    assert label_result.generated_stochastic_by_target.shape == (1, 4, 6, 4)


def test_skeleton_local_refinement_benchmark_smoke(capsys):
    data_config = DataConfig(kind="clifford", sigma_data=0.08, label_strategy="balanced")
    centers = centers_for_config(data_config, device="cpu")
    names = center_names_for_config(data_config)
    near_benchmarks = run_near_clifford_two_entangler_benchmark(
        clifford_gates=centers,
        clifford_labels=names,
        generated_gates=centers,
        generated_labels=names,
        n_targets=1,
        perturb_scale=0.05,
        n_random_candidates=16,
        n_analytic_gates=8,
        n_haar_gates=8,
        top_k=1,
        seed=9,
        keep_fidelities=False,
    )
    config = CircuitExperimentConfig(
        name="tiny-skeleton-local",
        schedule=DiffusionSchedule(T=4),
        train=CircuitTrainConfig(batch_size=4, num_steps=1, hidden=16, n_terms=4),
        sample_count=4,
    )
    target_only = run_target_conditioned_synthetic_circuit_experiment(
        torch.stack([item.target.unitary for item in near_benchmarks]),
        config,
        device="cpu",
        show_progress=False,
    )
    target_reports = run_target_conditioned_circuit_proposal_benchmark(
        near_benchmarks,
        target_only.generated_stochastic_by_target,
        top_k=1,
        keep_fidelities=False,
    )
    random_reports = run_joint_circuit_proposal_benchmark(
        near_benchmarks,
        q_normalized_random_stacks(8),
        top_k=1,
        keep_fidelities=False,
    )
    result = run_skeleton_local_refinement_benchmark(
        near_benchmarks,
        config,
        device="cpu",
        show_progress=False,
        top_k=1,
        init_noise_scale=0.01,
    )
    print_skeleton_local_refinement_summary(
        near_benchmarks,
        random_reports,
        target_reports,
        result.global_reports,
        result.local_reports,
    )

    captured = capsys.readouterr().out
    assert "skeleton-local diffusion" in captured
    assert result.generated_global_by_target.shape == (1, 4, 6, 4)
    assert result.generated_local_by_target.shape == (1, 4, 6, 4)


def q_normalized_random_stacks(n: int) -> torch.Tensor:
    q = torch.randn(n, 6, 4)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
