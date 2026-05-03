from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SynthesisCandidate:
    target: str
    template: str
    entangler: str
    fidelity: float
    slot_indices: tuple[int, int, int, int]
    slot_labels: tuple[str | None, str | None, str | None, str | None]


def quaternion_to_unitary(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q = q.to(torch.complex64)
    w, x, y, z = q.unbind(dim=-1)

    row0 = torch.stack([w - 1j * z, -y - 1j * x], dim=-1)
    row1 = torch.stack([y - 1j * x, w + 1j * z], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def kron2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.kron(a, b)


def local_layer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return kron2(a, b)


def two_qubit_gate(name: str, device: torch.device | str | None = None) -> torch.Tensor:
    name = name.lower()
    if name == "cnot":
        matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    elif name == "cz":
        matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ]
    elif name == "swap":
        matrix = [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    else:
        raise ValueError(f"Unknown two-qubit gate {name!r}")
    return torch.tensor(matrix, dtype=torch.complex64, device=device)


def bell_state(device: torch.device | str | None = None) -> torch.Tensor:
    state = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex64, device=device)
    return state / torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=device))


def compose_local_entangler_local(
    left_a: torch.Tensor,
    left_b: torch.Tensor,
    entangler: torch.Tensor,
    right_a: torch.Tensor,
    right_b: torch.Tensor,
) -> torch.Tensor:
    return local_layer(left_a, left_b) @ entangler @ local_layer(right_a, right_b)


def unitary_fidelity(candidate: torch.Tensor, target: torch.Tensor) -> float:
    dim = candidate.shape[-1]
    overlap = torch.trace(target.conj().T @ candidate).abs() / dim
    return overlap.real.item()


def bell_state_fidelity(candidate: torch.Tensor) -> float:
    initial = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64, device=candidate.device)
    target = bell_state(device=candidate.device)
    output = candidate @ initial
    return (target.conj() @ output).abs().square().real.item()


def synthesize_named_gate(
    local_gates: torch.Tensor,
    target: str = "cz",
    entangler: str = "cz",
    n_candidates: int = 1024,
    top_k: int = 5,
    local_labels: list[str | None] | None = None,
    slot_label_names: tuple[str, str, str, str] | None = None,
    seed: int = 0,
) -> list[SynthesisCandidate]:
    if local_gates.shape[0] == 0:
        raise ValueError("synthesize_named_gate needs at least one local gate")
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    target = target.lower()
    entangler = entangler.lower()
    device = local_gates.device
    units = quaternion_to_unitary(local_gates)
    target_unitary = two_qubit_gate(target, device=device)
    entangler_unitary = two_qubit_gate(entangler, device=device)

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = _sample_slot_indices(
        n_local=local_gates.shape[0],
        n_candidates=n_candidates,
        local_labels=local_labels,
        slot_label_names=slot_label_names,
        generator=generator,
    )

    candidates = []
    for slots in indices:
        left_a, left_b, right_a, right_b = (units[i] for i in slots)
        unitary = compose_local_entangler_local(left_a, left_b, entangler_unitary, right_a, right_b)
        labels = _labels_for_slots(slots, local_labels)
        candidates.append(
            SynthesisCandidate(
                target=target,
                template="local-entangler-local",
                entangler=entangler,
                fidelity=unitary_fidelity(unitary, target_unitary),
                slot_indices=tuple(slots),
                slot_labels=labels,
            )
        )

    candidates.sort(key=lambda candidate: candidate.fidelity, reverse=True)
    return candidates[:top_k]


def synthesize_bell_state(
    local_gates: torch.Tensor,
    entangler: str = "cnot",
    n_candidates: int = 1024,
    top_k: int = 5,
    local_labels: list[str | None] | None = None,
    slot_label_names: tuple[str, str, str, str] | None = None,
    seed: int = 0,
) -> list[SynthesisCandidate]:
    if local_gates.shape[0] == 0:
        raise ValueError("synthesize_bell_state needs at least one local gate")

    entangler = entangler.lower()
    device = local_gates.device
    units = quaternion_to_unitary(local_gates)
    entangler_unitary = two_qubit_gate(entangler, device=device)

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = _sample_slot_indices(
        n_local=local_gates.shape[0],
        n_candidates=n_candidates,
        local_labels=local_labels,
        slot_label_names=slot_label_names,
        generator=generator,
    )

    candidates = []
    for slots in indices:
        left_a, left_b, right_a, right_b = (units[i] for i in slots)
        unitary = compose_local_entangler_local(left_a, left_b, entangler_unitary, right_a, right_b)
        labels = _labels_for_slots(slots, local_labels)
        candidates.append(
            SynthesisCandidate(
                target="bell",
                template="local-entangler-local",
                entangler=entangler,
                fidelity=bell_state_fidelity(unitary),
                slot_indices=tuple(slots),
                slot_labels=labels,
            )
        )

    candidates.sort(key=lambda candidate: candidate.fidelity, reverse=True)
    return candidates[:top_k]


def print_synthesis_candidates(candidates: list[SynthesisCandidate]) -> None:
    header = "rank target   entangler fidelity   slot labels"
    print(header)
    print("-" * len(header))
    for rank, candidate in enumerate(candidates, start=1):
        labels = ", ".join(label if label is not None else "?" for label in candidate.slot_labels)
        print(
            f"{rank:<4} {candidate.target:<8} {candidate.entangler:<9} "
            f"{candidate.fidelity:>8.4f}   {labels}"
        )


def slot_labels_for_named_target(target: str, entangler: str) -> tuple[str, str, str, str]:
    target = target.lower()
    entangler = entangler.lower()
    if target == entangler:
        return ("I", "I", "I", "I")
    if {target, entangler} == {"cnot", "cz"}:
        return ("I", "H", "I", "H")
    raise ValueError(f"No built-in local-slot labels for target={target!r}, entangler={entangler!r}")


def _sample_slot_indices(
    n_local: int,
    n_candidates: int,
    local_labels: list[str | None] | None,
    slot_label_names: tuple[str, str, str, str] | None,
    generator: torch.Generator,
) -> list[list[int]]:
    if slot_label_names is None:
        return torch.randint(0, n_local, (n_candidates, 4), generator=generator).tolist()
    if local_labels is None:
        raise ValueError("slot_label_names requires local_labels")

    pools = []
    for label_name in slot_label_names:
        pool = [i for i, local_label in enumerate(local_labels) if local_label == label_name]
        if not pool:
            raise ValueError(f"No local gates found for requested slot label {label_name!r}")
        pools.append(pool)

    indices = []
    for _ in range(n_candidates):
        slots = []
        for pool in pools:
            choice = torch.randint(0, len(pool), (1,), generator=generator).item()
            slots.append(pool[choice])
        indices.append(slots)
    return indices


def _labels_for_slots(slots: list[int], local_labels: list[str | None] | None) -> tuple[str | None, str | None, str | None, str | None]:
    if local_labels is None:
        return (None, None, None, None)
    return tuple(local_labels[i] for i in slots)
