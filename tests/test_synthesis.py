import torch

from su2diffusion.synthesis import (
    bell_state_fidelity,
    compose_local_entangler_local,
    local_layer,
    quaternion_to_unitary,
    slot_labels_for_named_target,
    synthesize_bell_state,
    synthesize_named_gate,
    synthesize_named_gate_label_grid,
    synthesize_named_gate_unconstrained,
    two_qubit_gate,
    unitary_fidelity,
    unitary_fidelity_batch,
)


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


def test_bell_synthesis_returns_sorted_candidates():
    local_gates = torch.randn(16, 4)

    candidates = synthesize_bell_state(local_gates, entangler="cnot", n_candidates=16, top_k=5)

    assert len(candidates) == 5
    assert all(0.0 <= candidate.fidelity <= 1.0 for candidate in candidates)
