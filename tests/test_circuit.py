import torch

from su2diffusion.circuit import (
    CircuitDenoiser,
    CircuitExperimentConfig,
    CircuitTrainConfig,
    circuit_forward_heat_target,
    get_circuit_experiment_config,
    print_joint_circuit_comparison_summary,
    run_circuit_experiment,
    run_joint_circuit_proposal_benchmark,
    sample_circuit_reverse,
    sample_near_clifford_circuit_stacks,
    synthesize_unitary_from_circuit_stack_report,
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


def test_circuit_reverse_sampler_returns_normalized_stacks():
    schedule = DiffusionSchedule(T=4)
    model = CircuitDenoiser(T=schedule.T, hidden=16)

    q_stack = sample_circuit_reverse(model, schedule, n_samples=6, eta=0.0, device="cpu")

    assert q_stack.shape == (6, 6, 4)
    assert torch.isfinite(q_stack).all()
    assert torch.allclose(q_stack.norm(dim=-1), torch.ones(6, 6), atol=1e-5)


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
