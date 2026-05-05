import torch

from su2diffusion.data import center_names_for_config, centers_for_config, DataConfig
from su2diffusion.hamiltonian import (
    HamiltonianStackPredictor,
    HamiltonianPriorTrainConfig,
    HamiltonianSupervisedTrainConfig,
    evaluate_hamiltonian_stack_predictor,
    generate_hamiltonian_solution_dataset,
    hamiltonian_from_terms,
    hamiltonian_target_features,
    make_hamiltonian_target,
    make_random_pauli_hamiltonian_targets,
    parse_pauli_string,
    pauli_string_matrix,
    print_hamiltonian_prior_mixture_summary,
    print_hamiltonian_prior_search,
    print_hamiltonian_prior_search_summary,
    print_hamiltonian_target,
    print_hamiltonian_solution_dataset,
    print_hamiltonian_solution_dataset_summary,
    print_hamiltonian_seed_ablation,
    print_hamiltonian_seed_ablation_summary,
    print_hamiltonian_supervised_summary,
    print_hamiltonian_supervised_split_summary,
    print_hamiltonian_suite,
    print_hamiltonian_suite_summary,
    print_hamiltonian_two_entangler_benchmark,
    print_hamiltonian_two_entangler_summary,
    run_hamiltonian_prior_mixture_sweep,
    run_hamiltonian_prior_search_benchmark,
    run_hamiltonian_seed_ablation,
    run_hamiltonian_supervised_baseline,
    run_hamiltonian_supervised_split_baseline,
    run_hamiltonian_suite_benchmark,
    run_hamiltonian_two_entangler_benchmark,
    summarize_hamiltonian_suite,
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

    captured = capsys.readouterr().out
    assert "learned prior" in captured
    assert "alpha" in captured
    assert len(prior.losses) == 2
    assert 0.0 <= prior.train_accuracy <= 1.0
    assert len(benchmark.benchmarks) == 2
    assert set(mixture.alpha_results) == {0.0, 1.0}
    assert all(item.prior_report.candidates for item in benchmark.benchmarks)
    assert all(0.0 <= item.prior_report.candidates[0].fidelity <= 1.0 for item in benchmark.benchmarks)
