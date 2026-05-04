import torch

from su2diffusion.data import center_names_for_config, centers_for_config, DataConfig
from su2diffusion.hamiltonian import (
    hamiltonian_from_terms,
    make_hamiltonian_target,
    parse_pauli_string,
    pauli_string_matrix,
    print_hamiltonian_target,
    print_hamiltonian_two_entangler_benchmark,
    print_hamiltonian_two_entangler_summary,
    run_hamiltonian_two_entangler_benchmark,
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
