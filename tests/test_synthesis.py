import torch

from su2diffusion.synthesis import (
    SynthesisCandidate,
    bell_state_fidelity,
    compose_local_entangler_local,
    compose_two_entangler_local,
    local_layer,
    make_synthesis_report,
    make_hidden_shallow_circuit_targets,
    make_hidden_two_entangler_circuit_targets,
    print_hidden_shallow_circuit_benchmark,
    print_hidden_shallow_circuit_summary,
    print_hidden_two_entangler_circuit_benchmark,
    print_hidden_two_entangler_circuit_summary,
    print_refinement_ablation_results,
    print_refinement_ablation_summary,
    print_refinement_results,
    print_refinement_summary,
    print_synthesis_summary,
    quaternion_to_unitary,
    refine_hidden_two_entangler_benchmark,
    refine_two_entangler_candidate,
    run_refinement_ablation_benchmark,
    run_hidden_shallow_circuit_benchmark,
    run_hidden_two_entangler_circuit_benchmark,
    slot_labels_for_named_target,
    synthesize_bell_state,
    synthesize_bell_state_report,
    synthesize_named_gate,
    synthesize_named_gate_label_grid,
    synthesize_named_gate_label_grid_report,
    synthesize_named_gate_report,
    synthesize_named_gate_unconstrained,
    synthesize_named_gate_unconstrained_report,
    synthesize_unitary_label_grid_report,
    synthesize_unitary_two_entangler_random_report,
    synthesize_unitary_unconstrained_report,
    summarize_hidden_shallow_circuit_benchmark,
    summarize_hidden_two_entangler_circuit_benchmark,
    two_qubit_gate,
    unitary_fidelity,
    unitary_fidelity_batch,
)
from su2diffusion.quaternion import q_exp, q_mul, q_normalize


def _h_quaternion() -> torch.Tensor:
    inv_sqrt2 = 2.0**-0.5
    return torch.tensor([0.0, inv_sqrt2, 0.0, inv_sqrt2])


def test_quaternion_to_unitary_preserves_su2_structure():
    q = torch.randn(8, 4)

    unitary = quaternion_to_unitary(q)
    identity = torch.eye(2, dtype=torch.complex64)

    assert unitary.shape == (8, 2, 2)
    assert torch.allclose(unitary.conj().transpose(-1, -2) @ unitary, identity.expand(8, 2, 2), atol=1e-5)
    assert torch.allclose(torch.linalg.det(unitary), torch.ones(8, dtype=torch.complex64), atol=1e-5)


def test_known_quaternions_match_paulis_up_to_global_phase():
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    x_unitary = quaternion_to_unitary(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    y_unitary = quaternion_to_unitary(torch.tensor([0.0, 0.0, 1.0, 0.0]))
    z_unitary = quaternion_to_unitary(torch.tensor([0.0, 0.0, 0.0, 1.0]))

    assert unitary_fidelity(x_unitary, pauli_x) > 1.0 - 1e-6
    assert unitary_fidelity(y_unitary, pauli_y) > 1.0 - 1e-6
    assert unitary_fidelity(z_unitary, pauli_z) > 1.0 - 1e-6


def test_two_qubit_targets_and_bell_state_are_correct():
    h = quaternion_to_unitary(_h_quaternion())
    candidate = two_qubit_gate("cnot") @ local_layer(h, torch.eye(2, dtype=torch.complex64))

    assert unitary_fidelity(two_qubit_gate("cz"), two_qubit_gate("cz")) > 1.0 - 1e-6
    assert bell_state_fidelity(candidate) > 1.0 - 1e-6


def test_two_entangler_composition_matches_manual_product():
    h = quaternion_to_unitary(_h_quaternion())
    identity = torch.eye(2, dtype=torch.complex64)
    cz = two_qubit_gate("cz")

    actual = compose_two_entangler_local(identity, h, cz, h, identity, identity, h)
    expected = local_layer(identity, h) @ cz @ local_layer(h, identity) @ cz @ local_layer(identity, h)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_unitary_fidelity_is_global_phase_invariant():
    target = two_qubit_gate("cz")
    phased = 1j * target

    assert unitary_fidelity(phased, target) > 1.0 - 1e-6


def test_unitary_fidelity_batch_matches_scalar_fidelity():
    target = two_qubit_gate("cz")
    candidates = torch.stack([target, 1j * target])

    actual = unitary_fidelity_batch(candidates, target)

    assert torch.allclose(actual, torch.ones(2), atol=1e-6)


def test_named_gate_synthesis_can_use_label_constrained_slots():
    local_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]

    candidates = synthesize_named_gate(
        local_gates,
        target="cnot",
        entangler="cz",
        n_candidates=4,
        top_k=2,
        local_labels=labels,
        slot_label_names=slot_labels_for_named_target("cnot", "cz"),
    )

    assert len(candidates) == 2
    assert candidates[0].fidelity > 1.0 - 1e-6
    assert candidates[0].slot_labels == ("I", "H", "I", "H")
    assert candidates[0].fidelity >= candidates[1].fidelity


def test_guided_named_gate_report_keeps_full_fidelity_distribution():
    local_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]

    report = synthesize_named_gate_report(
        local_gates,
        target="cnot",
        entangler="cz",
        n_candidates=7,
        top_k=3,
        local_labels=labels,
        slot_label_names=slot_labels_for_named_target("cnot", "cz"),
        name="CNOT guided",
    )

    assert report.name == "CNOT guided"
    assert report.mode == "guided"
    assert len(report.candidates) == 3
    assert len(report.fidelities) == 7
    assert report.candidates[0].slot_labels == ("I", "H", "I", "H")


def test_synthesis_ranking_returns_sorted_finite_candidates():
    local_gates = torch.randn(16, 4)

    candidates = synthesize_named_gate(local_gates, target="cz", entangler="cz", n_candidates=16, top_k=5)

    assert len(candidates) == 5
    assert all(0.0 <= candidate.fidelity <= 1.0 for candidate in candidates)
    assert [candidate.fidelity for candidate in candidates] == sorted(
        [candidate.fidelity for candidate in candidates],
        reverse=True,
    )


def test_unconstrained_named_gate_synthesis_finds_known_decomposition():
    local_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]

    candidates = synthesize_named_gate_unconstrained(
        local_gates,
        target="cnot",
        entangler="cz",
        n_candidates=128,
        top_k=3,
        local_labels=labels,
        seed=2,
    )

    assert len(candidates) == 3
    assert candidates[0].fidelity > 1.0 - 1e-6
    assert candidates[0].slot_labels == ("I", "H", "I", "H")


def test_label_grid_synthesis_finds_known_decompositions():
    local_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]

    cz = synthesize_named_gate_label_grid(local_gates, labels, target="cz", entangler="cz", top_k=1)
    cnot = synthesize_named_gate_label_grid(local_gates, labels, target="cnot", entangler="cz", top_k=1)

    assert cz[0].fidelity > 1.0 - 1e-6
    assert cz[0].slot_labels == ("I", "I", "I", "I")
    assert cnot[0].fidelity > 1.0 - 1e-6
    assert cnot[0].slot_labels == ("I", "H", "I", "H")


def test_label_grid_report_keeps_full_fidelity_distribution():
    local_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]

    report = synthesize_named_gate_label_grid_report(
        local_gates,
        labels,
        target="cnot",
        entangler="cz",
        top_k=2,
        name="CNOT label grid",
    )

    assert report.name == "CNOT label grid"
    assert report.mode == "label-grid"
    assert len(report.candidates) == 2
    assert len(report.fidelities) == 16
    assert report.candidates[0].fidelity >= report.candidates[1].fidelity


def test_random_search_report_keeps_full_fidelity_distribution():
    local_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]

    report = synthesize_named_gate_unconstrained_report(
        local_gates,
        target="cnot",
        entangler="cz",
        n_candidates=32,
        top_k=3,
        local_labels=labels,
        seed=2,
    )

    assert report.mode == "random"
    assert len(report.candidates) == 3
    assert len(report.fidelities) == 32
    assert report.candidates[0].fidelity >= report.candidates[1].fidelity


def test_synthesis_summary_prints_report_statistics(capsys):
    local_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    candidates = synthesize_named_gate(
        local_gates,
        target="cnot",
        entangler="cz",
        n_candidates=4,
        top_k=2,
        slot_label_names=None,
    )
    report = make_synthesis_report(candidates, name="guided", mode="guided")

    print_synthesis_summary([report])

    captured = capsys.readouterr().out
    assert "guided" in captured
    assert "top-k mean" in captured


def test_bell_synthesis_returns_sorted_candidates():
    local_gates = torch.randn(16, 4)

    candidates = synthesize_bell_state(local_gates, entangler="cnot", n_candidates=16, top_k=5)

    assert len(candidates) == 5
    assert all(0.0 <= candidate.fidelity <= 1.0 for candidate in candidates)


def test_bell_synthesis_report_keeps_full_fidelity_distribution():
    local_gates = torch.randn(16, 4)

    report = synthesize_bell_state_report(local_gates, entangler="cnot", n_candidates=16, top_k=5)

    assert len(report.candidates) == 5
    assert len(report.fidelities) == 16


def test_unitary_label_grid_report_finds_hidden_template_target():
    local_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]
    h = quaternion_to_unitary(_h_quaternion())
    target = compose_local_entangler_local(
        torch.eye(2, dtype=torch.complex64),
        h,
        two_qubit_gate("cz"),
        torch.eye(2, dtype=torch.complex64),
        h,
    )

    report = synthesize_unitary_label_grid_report(
        local_gates,
        labels,
        target_unitary=target,
        target_name="hidden",
        entangler="cz",
        top_k=1,
    )

    assert report.candidates[0].fidelity > 1.0 - 1e-6
    assert report.candidates[0].slot_labels == ("I", "H", "I", "H")


def test_unitary_random_report_returns_finite_candidates():
    local_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]
    target = two_qubit_gate("cz")

    report = synthesize_unitary_unconstrained_report(
        local_gates,
        target_unitary=target,
        target_name="hidden",
        entangler="cz",
        n_candidates=32,
        top_k=3,
        local_labels=labels,
    )

    assert len(report.candidates) == 3
    assert len(report.fidelities) == 32
    assert all(0.0 <= candidate.fidelity <= 1.0 for candidate in report.candidates)


def test_hidden_shallow_circuit_benchmark_recovers_exact_targets(capsys):
    exact_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]

    targets = make_hidden_shallow_circuit_targets(exact_gates, labels, n_targets=2, seed=4)
    benchmarks = run_hidden_shallow_circuit_benchmark(
        exact_gates=exact_gates,
        exact_labels=labels,
        generated_gates=exact_gates,
        generated_labels=labels,
        n_targets=2,
        n_random_candidates=32,
        top_k=2,
        seed=4,
    )
    print_hidden_shallow_circuit_benchmark(benchmarks)

    captured = capsys.readouterr().out
    assert "hidden labels" in captured
    assert len(targets) == 2
    assert len(benchmarks) == 2
    assert all(benchmark.exact_report.candidates[0].fidelity > 1.0 - 1e-6 for benchmark in benchmarks)
    assert all(benchmark.generated_label_grid_report.candidates[0].fidelity > 1.0 - 1e-6 for benchmark in benchmarks)


def test_hidden_shallow_circuit_benchmark_summary(capsys):
    exact_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]

    benchmarks = run_hidden_shallow_circuit_benchmark(
        exact_gates=exact_gates,
        exact_labels=labels,
        generated_gates=exact_gates,
        generated_labels=labels,
        n_targets=3,
        n_random_candidates=32,
        top_k=2,
        seed=8,
        keep_fidelities=False,
    )
    aggregates = summarize_hidden_shallow_circuit_benchmark(benchmarks)
    print_hidden_shallow_circuit_summary(aggregates)

    captured = capsys.readouterr().out
    assert "generated-label-grid" in captured
    assert len(aggregates) == 3
    assert aggregates[0].n_targets == 3
    assert aggregates[0].success_99 == 1.0
    assert all(len(item.generated_label_grid_report.fidelities) == 2 for item in benchmarks)


def test_two_entangler_random_report_finds_hidden_candidate():
    local_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]
    units = quaternion_to_unitary(local_gates)
    target = compose_two_entangler_local(
        units[0],
        units[1],
        two_qubit_gate("cz"),
        units[0],
        units[1],
        units[0],
        units[1],
    )

    report = synthesize_unitary_two_entangler_random_report(
        local_gates,
        target,
        target_name="depth2",
        entangler="cz",
        n_candidates=256,
        top_k=3,
        local_labels=labels,
        seed=1,
    )

    assert len(report.candidates) == 3
    assert len(report.candidates[0].slot_indices) == 6
    assert all(0.0 <= candidate.fidelity <= 1.0 for candidate in report.candidates)


def test_hidden_two_entangler_benchmark_summary(capsys):
    exact_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]

    targets = make_hidden_two_entangler_circuit_targets(exact_gates, labels, n_targets=2, seed=3)
    benchmarks = run_hidden_two_entangler_circuit_benchmark(
        exact_gates=exact_gates,
        exact_labels=labels,
        generated_gates=exact_gates,
        generated_labels=labels,
        n_targets=2,
        n_random_candidates=128,
        top_k=2,
        seed=3,
        keep_fidelities=False,
    )
    print_hidden_two_entangler_circuit_benchmark(benchmarks)
    aggregates = summarize_hidden_two_entangler_circuit_benchmark(benchmarks)
    print_hidden_two_entangler_circuit_summary(aggregates)

    captured = capsys.readouterr().out
    assert "exact-random" in captured
    assert len(targets) == 2
    assert len(benchmarks) == 2
    assert len(benchmarks[0].target.slot_labels) == 6
    assert aggregates[0].mode == "oracle"
    assert aggregates[0].success_99 == 1.0


def test_two_entangler_refinement_improves_perturbed_candidate(capsys):
    exact_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    units = quaternion_to_unitary(exact_gates)
    target = compose_two_entangler_local(
        units[0],
        units[1],
        two_qubit_gate("cz"),
        units[0],
        units[1],
        units[0],
        units[1],
    )
    perturb = torch.tensor(
        [
            [0.08, 0.01, -0.02],
            [-0.03, 0.05, 0.01],
        ]
    )
    generated_gates = q_normalize(q_mul(q_exp(perturb), exact_gates))
    generated_units = quaternion_to_unitary(generated_gates)
    candidate_unitary = compose_two_entangler_local(
        generated_units[0],
        generated_units[1],
        two_qubit_gate("cz"),
        generated_units[0],
        generated_units[1],
        generated_units[0],
        generated_units[1],
    )
    candidate = SynthesisCandidate(
        target="depth2",
        template="two-entangler-local",
        entangler="cz",
        fidelity=unitary_fidelity(candidate_unitary, target),
        slot_indices=(0, 1, 0, 1, 0, 1),
        slot_labels=("I", "H", "I", "H", "I", "H"),
    )

    result = refine_two_entangler_candidate(
        generated_gates,
        candidate,
        target,
        num_steps=40,
        lr=0.05,
    )
    print_refinement_results([result])
    print_refinement_summary([result])

    captured = capsys.readouterr().out
    assert "before" in captured
    assert result.refined_gates.shape == (6, 4)
    assert result.refined_fidelity > result.initial_fidelity + 1e-3
    assert result.refined_fidelity > 0.99


def test_refine_hidden_two_entangler_benchmark_uses_generated_report_candidate():
    exact_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]
    benchmarks = run_hidden_two_entangler_circuit_benchmark(
        exact_gates=exact_gates,
        exact_labels=labels,
        generated_gates=exact_gates,
        generated_labels=labels,
        n_targets=1,
        n_random_candidates=128,
        top_k=1,
        seed=3,
        keep_fidelities=False,
    )

    results = refine_hidden_two_entangler_benchmark(
        benchmarks,
        generated_gates=exact_gates,
        num_steps=5,
        lr=0.02,
    )

    assert len(results) == 1
    assert results[0].refined_fidelity >= results[0].initial_fidelity - 1e-5


def test_refinement_ablation_compares_generated_and_random_starts(capsys):
    exact_gates = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            _h_quaternion(),
        ]
    )
    labels = ["I", "H"]
    benchmarks = run_hidden_two_entangler_circuit_benchmark(
        exact_gates=exact_gates,
        exact_labels=labels,
        generated_gates=exact_gates,
        generated_labels=labels,
        n_targets=1,
        n_random_candidates=128,
        top_k=1,
        seed=3,
        keep_fidelities=False,
    )
    generated_results = refine_hidden_two_entangler_benchmark(
        benchmarks,
        generated_gates=exact_gates,
        num_steps=5,
        lr=0.02,
    )

    ablations = run_refinement_ablation_benchmark(
        benchmarks,
        generated_gates=exact_gates,
        generated_results=generated_results,
        n_random_starts=2,
        num_steps=5,
        lr=0.02,
        seed=1,
    )
    print_refinement_ablation_results(ablations)
    print_refinement_ablation_summary(ablations)

    captured = capsys.readouterr().out
    assert "gen before" in captured
    assert len(ablations) == 1
    assert 0.0 <= ablations[0].random.initial_fidelity <= 1.0
    assert 0.0 <= ablations[0].random.refined_fidelity <= 1.0
