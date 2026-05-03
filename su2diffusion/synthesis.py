from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch


@dataclass(frozen=True)
class SynthesisCandidate:
    target: str
    template: str
    entangler: str
    fidelity: float
    slot_indices: tuple[int, ...]
    slot_labels: tuple[str | None, ...]


@dataclass(frozen=True)
class SynthesisReport:
    name: str
    mode: str
    target: str
    entangler: str
    candidates: list[SynthesisCandidate]
    fidelities: tuple[float, ...]


@dataclass(frozen=True)
class HiddenShallowCircuitTarget:
    name: str
    entangler: str
    unitary: torch.Tensor
    slot_indices: tuple[int, ...]
    slot_labels: tuple[str, ...]


@dataclass(frozen=True)
class HiddenShallowCircuitBenchmark:
    target: HiddenShallowCircuitTarget
    exact_report: SynthesisReport
    generated_label_grid_report: SynthesisReport
    random_report: SynthesisReport


@dataclass(frozen=True)
class HiddenTwoEntanglerCircuitBenchmark:
    target: HiddenShallowCircuitTarget
    oracle_fidelity: float
    exact_random_report: SynthesisReport
    generated_random_report: SynthesisReport


@dataclass(frozen=True)
class HiddenShallowCircuitAggregate:
    mode: str
    n_targets: int
    mean_best: float
    median_best: float
    min_best: float
    max_best: float
    success_95: float
    success_98: float
    success_99: float


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


def compose_two_entangler_local(
    first_a: torch.Tensor,
    first_b: torch.Tensor,
    entangler: torch.Tensor,
    middle_a: torch.Tensor,
    middle_b: torch.Tensor,
    second_a: torch.Tensor,
    second_b: torch.Tensor,
) -> torch.Tensor:
    return (
        local_layer(first_a, first_b)
        @ entangler
        @ local_layer(middle_a, middle_b)
        @ entangler
        @ local_layer(second_a, second_b)
    )


def unitary_fidelity(candidate: torch.Tensor, target: torch.Tensor) -> float:
    dim = candidate.shape[-1]
    overlap = torch.trace(target.conj().T @ candidate).abs() / dim
    return overlap.real.clamp(0.0, 1.0).item()


def unitary_fidelity_batch(candidates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dim = candidates.shape[-1]
    overlaps = torch.einsum("ij,nij->n", target.conj(), candidates).abs() / dim
    return overlaps.real.clamp(0.0, 1.0)


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
    return synthesize_named_gate_report(
        local_gates,
        target=target,
        entangler=entangler,
        n_candidates=n_candidates,
        top_k=top_k,
        local_labels=local_labels,
        slot_label_names=slot_label_names,
        seed=seed,
    ).candidates


def synthesize_named_gate_report(
    local_gates: torch.Tensor,
    target: str = "cz",
    entangler: str = "cz",
    n_candidates: int = 1024,
    top_k: int = 5,
    local_labels: list[str | None] | None = None,
    slot_label_names: tuple[str, str, str, str] | None = None,
    seed: int = 0,
    name: str | None = None,
) -> SynthesisReport:
    if local_gates.shape[0] == 0:
        raise ValueError("synthesize_named_gate_report needs at least one local gate")
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    target = target.lower()
    entangler = entangler.lower()
    device = local_gates.device
    target_unitary = two_qubit_gate(target, device=device)

    return synthesize_unitary_guided_report(
        local_gates,
        target_unitary=target_unitary,
        target_name=target,
        entangler=entangler,
        n_candidates=n_candidates,
        top_k=top_k,
        local_labels=local_labels,
        slot_label_names=slot_label_names,
        seed=seed,
        name=name or f"{target.upper()} guided search",
    )


def synthesize_unitary_guided_report(
    local_gates: torch.Tensor,
    target_unitary: torch.Tensor,
    target_name: str = "target",
    entangler: str = "cz",
    n_candidates: int = 1024,
    top_k: int = 5,
    local_labels: list[str | None] | None = None,
    slot_label_names: tuple[str, str, str, str] | None = None,
    seed: int = 0,
    name: str | None = None,
) -> SynthesisReport:
    if local_gates.shape[0] == 0:
        raise ValueError("synthesize_unitary_guided_report needs at least one local gate")
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    target_name = target_name.lower()
    entangler = entangler.lower()
    device = local_gates.device
    units = quaternion_to_unitary(local_gates)
    target_unitary = target_unitary.to(device=device, dtype=torch.complex64)
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
                target=target_name,
                template="local-entangler-local",
                entangler=entangler,
                fidelity=unitary_fidelity(unitary, target_unitary),
                slot_indices=tuple(slots),
                slot_labels=labels,
            )
        )

    candidates.sort(key=lambda candidate: candidate.fidelity, reverse=True)
    return make_synthesis_report(
        candidates[:top_k],
        name=name or f"{target_name} guided search",
        mode="guided",
        fidelities=[candidate.fidelity for candidate in candidates],
    )


def synthesize_named_gate_unconstrained(
    local_gates: torch.Tensor,
    target: str = "cz",
    entangler: str = "cz",
    n_candidates: int = 100_000,
    top_k: int = 5,
    local_labels: list[str | None] | None = None,
    seed: int = 0,
) -> list[SynthesisCandidate]:
    return synthesize_named_gate_unconstrained_report(
        local_gates,
        target=target,
        entangler=entangler,
        n_candidates=n_candidates,
        top_k=top_k,
        local_labels=local_labels,
        seed=seed,
    ).candidates


def synthesize_named_gate_label_grid(
    local_gates: torch.Tensor,
    local_labels: list[str],
    target: str = "cz",
    entangler: str = "cz",
    top_k: int = 5,
) -> list[SynthesisCandidate]:
    return synthesize_named_gate_label_grid_report(
        local_gates,
        local_labels,
        target=target,
        entangler=entangler,
        top_k=top_k,
    ).candidates


def synthesize_named_gate_unconstrained_report(
    local_gates: torch.Tensor,
    target: str = "cz",
    entangler: str = "cz",
    n_candidates: int = 100_000,
    top_k: int = 5,
    local_labels: list[str | None] | None = None,
    seed: int = 0,
    name: str | None = None,
) -> SynthesisReport:
    if local_gates.shape[0] == 0:
        raise ValueError("synthesize_named_gate_unconstrained_report needs at least one local gate")
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    target = target.lower()
    entangler = entangler.lower()
    device = local_gates.device
    target_unitary = two_qubit_gate(target, device=device)

    return synthesize_unitary_unconstrained_report(
        local_gates,
        target_unitary=target_unitary,
        target_name=target,
        entangler=entangler,
        n_candidates=n_candidates,
        top_k=top_k,
        local_labels=local_labels,
        seed=seed,
        name=name or f"{target.upper()} random generated search",
    )


def synthesize_named_gate_label_grid_report(
    local_gates: torch.Tensor,
    local_labels: list[str],
    target: str = "cz",
    entangler: str = "cz",
    top_k: int = 5,
    name: str | None = None,
) -> SynthesisReport:
    target = target.lower()
    entangler = entangler.lower()
    device = local_gates.device
    target_unitary = two_qubit_gate(target, device=device)

    return synthesize_unitary_label_grid_report(
        local_gates,
        local_labels,
        target_unitary=target_unitary,
        target_name=target,
        entangler=entangler,
        top_k=top_k,
        name=name or f"{target.upper()} label-grid search",
    )


def synthesize_unitary_unconstrained_report(
    local_gates: torch.Tensor,
    target_unitary: torch.Tensor,
    target_name: str = "target",
    entangler: str = "cz",
    n_candidates: int = 100_000,
    top_k: int = 5,
    local_labels: list[str | None] | None = None,
    seed: int = 0,
    name: str | None = None,
    keep_fidelities: bool = True,
) -> SynthesisReport:
    if local_gates.shape[0] == 0:
        raise ValueError("synthesize_unitary_unconstrained_report needs at least one local gate")
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    target_name = target_name.lower()
    entangler = entangler.lower()
    device = local_gates.device
    units = quaternion_to_unitary(local_gates)
    target_unitary = target_unitary.to(device=device, dtype=torch.complex64)
    entangler_unitary = two_qubit_gate(entangler, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    indices = torch.randint(
        low=0,
        high=local_gates.shape[0],
        size=(n_candidates, 4),
        device=device,
        generator=generator,
    )

    left = _batched_local_layer(units[indices[:, 0]], units[indices[:, 1]])
    right = _batched_local_layer(units[indices[:, 2]], units[indices[:, 3]])
    entanglers = entangler_unitary.expand(n_candidates, 4, 4)
    unitaries = left @ entanglers @ right
    fidelities = unitary_fidelity_batch(unitaries, target_unitary)
    candidates = _top_candidates_from_slots(
        fidelities=fidelities,
        slot_indices=indices,
        target=target_name,
        entangler=entangler,
        template="unconstrained-local-entangler-local",
        top_k=top_k,
        local_labels=local_labels,
    )
    return make_synthesis_report(
        candidates,
        name=name or f"{target_name} random generated search",
        mode="random",
        fidelities=fidelities.tolist() if keep_fidelities else None,
    )


def synthesize_unitary_two_entangler_random_report(
    local_gates: torch.Tensor,
    target_unitary: torch.Tensor,
    target_name: str = "target",
    entangler: str = "cz",
    n_candidates: int = 100_000,
    top_k: int = 5,
    local_labels: list[str | None] | None = None,
    seed: int = 0,
    name: str | None = None,
    keep_fidelities: bool = True,
) -> SynthesisReport:
    if local_gates.shape[0] == 0:
        raise ValueError("synthesize_unitary_two_entangler_random_report needs at least one local gate")
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    target_name = target_name.lower()
    entangler = entangler.lower()
    device = local_gates.device
    units = quaternion_to_unitary(local_gates)
    target_unitary = target_unitary.to(device=device, dtype=torch.complex64)
    entangler_unitary = two_qubit_gate(entangler, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    indices = torch.randint(
        low=0,
        high=local_gates.shape[0],
        size=(n_candidates, 6),
        device=device,
        generator=generator,
    )

    first = _batched_local_layer(units[indices[:, 0]], units[indices[:, 1]])
    middle = _batched_local_layer(units[indices[:, 2]], units[indices[:, 3]])
    second = _batched_local_layer(units[indices[:, 4]], units[indices[:, 5]])
    entanglers = entangler_unitary.expand(n_candidates, 4, 4)
    unitaries = first @ entanglers @ middle @ entanglers @ second
    fidelities = unitary_fidelity_batch(unitaries, target_unitary)
    candidates = _top_candidates_from_slots(
        fidelities=fidelities,
        slot_indices=indices,
        target=target_name,
        entangler=entangler,
        template="two-entangler-local",
        top_k=top_k,
        local_labels=local_labels,
    )
    return make_synthesis_report(
        candidates,
        name=name or f"{target_name} two-entangler random search",
        mode="two-entangler-random",
        fidelities=fidelities.tolist() if keep_fidelities else None,
    )


def synthesize_unitary_label_grid_report(
    local_gates: torch.Tensor,
    local_labels: list[str],
    target_unitary: torch.Tensor,
    target_name: str = "target",
    entangler: str = "cz",
    top_k: int = 5,
    name: str | None = None,
    keep_fidelities: bool = True,
) -> SynthesisReport:
    if local_gates.shape[0] == 0:
        raise ValueError("synthesize_unitary_label_grid_report needs at least one local gate")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    target_name = target_name.lower()
    entangler = entangler.lower()
    device = local_gates.device
    unique_labels = list(dict.fromkeys(local_labels))
    representative_indices = [_first_index_for_label(local_labels, label) for label in unique_labels]
    representatives = local_gates[representative_indices]
    units = quaternion_to_unitary(representatives)
    target_unitary = target_unitary.to(device=device, dtype=torch.complex64)
    entangler_unitary = two_qubit_gate(entangler, device=device)

    n_labels = len(unique_labels)
    grid = torch.cartesian_prod(
        torch.arange(n_labels, device=device),
        torch.arange(n_labels, device=device),
        torch.arange(n_labels, device=device),
        torch.arange(n_labels, device=device),
    )
    left = _batched_local_layer(units[grid[:, 0]], units[grid[:, 1]])
    right = _batched_local_layer(units[grid[:, 2]], units[grid[:, 3]])
    entanglers = entangler_unitary.expand(grid.shape[0], 4, 4)
    unitaries = left @ entanglers @ right
    fidelities = unitary_fidelity_batch(unitaries, target_unitary)
    values, rows = torch.topk(fidelities, k=min(top_k, grid.shape[0]))

    candidates = []
    for value, row in zip(values.tolist(), rows.tolist()):
        label_slots = grid[row].tolist()
        sample_slots = tuple(representative_indices[label_idx] for label_idx in label_slots)
        candidates.append(
            SynthesisCandidate(
                target=target_name,
                template="label-grid-local-entangler-local",
                entangler=entangler,
                fidelity=value,
                slot_indices=sample_slots,
                slot_labels=tuple(unique_labels[label_idx] for label_idx in label_slots),
            )
        )
    return make_synthesis_report(
        candidates,
        name=name or f"{target_name} label-grid search",
        mode="label-grid",
        fidelities=fidelities.tolist() if keep_fidelities else None,
    )


def synthesize_bell_state(
    local_gates: torch.Tensor,
    entangler: str = "cnot",
    n_candidates: int = 1024,
    top_k: int = 5,
    local_labels: list[str | None] | None = None,
    slot_label_names: tuple[str, str, str, str] | None = None,
    seed: int = 0,
) -> list[SynthesisCandidate]:
    return synthesize_bell_state_report(
        local_gates,
        entangler=entangler,
        n_candidates=n_candidates,
        top_k=top_k,
        local_labels=local_labels,
        slot_label_names=slot_label_names,
        seed=seed,
    ).candidates


def synthesize_bell_state_report(
    local_gates: torch.Tensor,
    entangler: str = "cnot",
    n_candidates: int = 1024,
    top_k: int = 5,
    local_labels: list[str | None] | None = None,
    slot_label_names: tuple[str, str, str, str] | None = None,
    seed: int = 0,
    name: str | None = None,
) -> SynthesisReport:
    if local_gates.shape[0] == 0:
        raise ValueError("synthesize_bell_state_report needs at least one local gate")

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
    return make_synthesis_report(
        candidates[:top_k],
        name=name or "Bell guided search",
        mode="guided",
        fidelities=[candidate.fidelity for candidate in candidates],
    )


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


def make_synthesis_report(
    candidates: list[SynthesisCandidate],
    name: str,
    mode: str,
    fidelities: list[float] | tuple[float, ...] | None = None,
) -> SynthesisReport:
    if not candidates:
        raise ValueError("make_synthesis_report needs at least one candidate")
    first = candidates[0]
    values = tuple(float(candidate.fidelity) for candidate in candidates) if fidelities is None else tuple(fidelities)
    return SynthesisReport(
        name=name,
        mode=mode,
        target=first.target,
        entangler=first.entangler,
        candidates=candidates,
        fidelities=values,
    )


def print_synthesis_summary(reports: list[SynthesisReport] | dict[str, SynthesisReport], top_n: int = 10) -> None:
    rows = list(reports.values()) if isinstance(reports, dict) else reports
    header = "name                         target   mode        best     median   top-k mean   best labels"
    print(header)
    print("-" * len(header))
    for report in rows:
        values = torch.tensor(report.fidelities, dtype=torch.float32)
        top_values = torch.topk(values, k=min(top_n, values.numel())).values
        labels = ", ".join(label if label is not None else "?" for label in report.candidates[0].slot_labels)
        print(
            f"{report.name:<28} {report.target:<8} {report.mode:<10} "
            f"{values.max().item():>7.4f}   {values.median().item():>7.4f}   "
            f"{top_values.mean().item():>10.4f}   {labels}"
        )


def plot_synthesis_fidelity_histograms(
    reports: list[SynthesisReport] | dict[str, SynthesisReport],
    bins: int = 50,
) -> None:
    rows = list(reports.values()) if isinstance(reports, dict) else reports
    if not rows:
        raise ValueError("plot_synthesis_fidelity_histograms needs at least one report")

    plt.figure(figsize=(8, 4))
    for report in rows:
        plt.hist(report.fidelities, bins=bins, alpha=0.45, density=True, label=report.name)
    plt.xlabel("unitary fidelity")
    plt.ylabel("density")
    plt.title("Synthesis search fidelity distributions")
    plt.legend()
    plt.tight_layout()


def make_hidden_shallow_circuit_targets(
    local_gates: torch.Tensor,
    local_labels: list[str],
    n_targets: int = 8,
    entangler: str = "cz",
    seed: int = 0,
) -> list[HiddenShallowCircuitTarget]:
    if local_gates.shape[0] == 0:
        raise ValueError("make_hidden_shallow_circuit_targets needs at least one local gate")
    if n_targets <= 0:
        raise ValueError("n_targets must be positive")

    entangler = entangler.lower()
    device = local_gates.device
    units = quaternion_to_unitary(local_gates)
    entangler_unitary = two_qubit_gate(entangler, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    slot_indices = torch.randint(
        low=0,
        high=local_gates.shape[0],
        size=(n_targets, 4),
        device=device,
        generator=generator,
    )

    targets = []
    for i, slots_tensor in enumerate(slot_indices):
        slots = tuple(int(index) for index in slots_tensor.tolist())
        left_a, left_b, right_a, right_b = (units[index] for index in slots)
        unitary = compose_local_entangler_local(left_a, left_b, entangler_unitary, right_a, right_b)
        targets.append(
            HiddenShallowCircuitTarget(
                name=f"hidden-{i:02d}",
                entangler=entangler,
                unitary=unitary,
                slot_indices=slots,
                slot_labels=tuple(local_labels[index] for index in slots),
            )
        )
    return targets


def make_hidden_two_entangler_circuit_targets(
    local_gates: torch.Tensor,
    local_labels: list[str],
    n_targets: int = 8,
    entangler: str = "cz",
    seed: int = 0,
) -> list[HiddenShallowCircuitTarget]:
    if local_gates.shape[0] == 0:
        raise ValueError("make_hidden_two_entangler_circuit_targets needs at least one local gate")
    if n_targets <= 0:
        raise ValueError("n_targets must be positive")

    entangler = entangler.lower()
    device = local_gates.device
    units = quaternion_to_unitary(local_gates)
    entangler_unitary = two_qubit_gate(entangler, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    slot_indices = torch.randint(
        low=0,
        high=local_gates.shape[0],
        size=(n_targets, 6),
        device=device,
        generator=generator,
    )

    targets = []
    for i, slots_tensor in enumerate(slot_indices):
        slots = tuple(int(index) for index in slots_tensor.tolist())
        first_a, first_b, middle_a, middle_b, second_a, second_b = (units[index] for index in slots)
        unitary = compose_two_entangler_local(
            first_a,
            first_b,
            entangler_unitary,
            middle_a,
            middle_b,
            second_a,
            second_b,
        )
        targets.append(
            HiddenShallowCircuitTarget(
                name=f"depth2-{i:02d}",
                entangler=entangler,
                unitary=unitary,
                slot_indices=slots,
                slot_labels=tuple(local_labels[index] for index in slots),
            )
        )
    return targets


def run_hidden_shallow_circuit_benchmark(
    exact_gates: torch.Tensor,
    exact_labels: list[str],
    generated_gates: torch.Tensor,
    generated_labels: list[str],
    n_targets: int = 8,
    entangler: str = "cz",
    n_random_candidates: int = 100_000,
    top_k: int = 5,
    seed: int = 0,
    keep_fidelities: bool = True,
) -> list[HiddenShallowCircuitBenchmark]:
    targets = make_hidden_shallow_circuit_targets(
        exact_gates,
        exact_labels,
        n_targets=n_targets,
        entangler=entangler,
        seed=seed,
    )

    benchmarks = []
    for i, target in enumerate(targets):
        exact_report = synthesize_unitary_label_grid_report(
            exact_gates,
            exact_labels,
            target_unitary=target.unitary,
            target_name=target.name,
            entangler=entangler,
            top_k=top_k,
            name=f"{target.name} exact grid",
            keep_fidelities=keep_fidelities,
        )
        generated_label_grid_report = synthesize_unitary_label_grid_report(
            generated_gates,
            generated_labels,
            target_unitary=target.unitary,
            target_name=target.name,
            entangler=entangler,
            top_k=top_k,
            name=f"{target.name} generated label grid",
            keep_fidelities=keep_fidelities,
        )
        random_report = synthesize_unitary_unconstrained_report(
            generated_gates,
            target_unitary=target.unitary,
            target_name=target.name,
            entangler=entangler,
            n_candidates=n_random_candidates,
            top_k=top_k,
            local_labels=generated_labels,
            seed=seed + i + 1,
            name=f"{target.name} generated random",
            keep_fidelities=keep_fidelities,
        )
        benchmarks.append(
            HiddenShallowCircuitBenchmark(
                target=target,
                exact_report=exact_report,
                generated_label_grid_report=generated_label_grid_report,
                random_report=random_report,
            )
        )
    return benchmarks


def run_hidden_two_entangler_circuit_benchmark(
    exact_gates: torch.Tensor,
    exact_labels: list[str],
    generated_gates: torch.Tensor,
    generated_labels: list[str],
    n_targets: int = 8,
    entangler: str = "cz",
    n_random_candidates: int = 200_000,
    top_k: int = 5,
    seed: int = 0,
    keep_fidelities: bool = True,
) -> list[HiddenTwoEntanglerCircuitBenchmark]:
    targets = make_hidden_two_entangler_circuit_targets(
        exact_gates,
        exact_labels,
        n_targets=n_targets,
        entangler=entangler,
        seed=seed,
    )

    benchmarks = []
    for i, target in enumerate(targets):
        exact_random_report = synthesize_unitary_two_entangler_random_report(
            exact_gates,
            target_unitary=target.unitary,
            target_name=target.name,
            entangler=entangler,
            n_candidates=n_random_candidates,
            top_k=top_k,
            local_labels=exact_labels,
            seed=seed + 10_000 + i,
            name=f"{target.name} exact random",
            keep_fidelities=keep_fidelities,
        )
        generated_random_report = synthesize_unitary_two_entangler_random_report(
            generated_gates,
            target_unitary=target.unitary,
            target_name=target.name,
            entangler=entangler,
            n_candidates=n_random_candidates,
            top_k=top_k,
            local_labels=generated_labels,
            seed=seed + 20_000 + i,
            name=f"{target.name} generated random",
            keep_fidelities=keep_fidelities,
        )
        benchmarks.append(
            HiddenTwoEntanglerCircuitBenchmark(
                target=target,
                oracle_fidelity=1.0,
                exact_random_report=exact_random_report,
                generated_random_report=generated_random_report,
            )
        )
    return benchmarks


def print_hidden_shallow_circuit_benchmark(
    benchmarks: list[HiddenShallowCircuitBenchmark],
) -> None:
    header = "target     hidden labels                    exact    gen-grid   random   best generated labels"
    print(header)
    print("-" * len(header))
    for benchmark in benchmarks:
        hidden_labels = ", ".join(benchmark.target.slot_labels)
        best_labels = ", ".join(
            label if label is not None else "?"
            for label in benchmark.generated_label_grid_report.candidates[0].slot_labels
        )
        print(
            f"{benchmark.target.name:<10} {hidden_labels:<32} "
            f"{benchmark.exact_report.candidates[0].fidelity:>6.4f}   "
            f"{benchmark.generated_label_grid_report.candidates[0].fidelity:>8.4f}   "
            f"{benchmark.random_report.candidates[0].fidelity:>6.4f}   "
            f"{best_labels}"
        )


def print_hidden_two_entangler_circuit_benchmark(
    benchmarks: list[HiddenTwoEntanglerCircuitBenchmark],
) -> None:
    header = "target     hidden labels                                      oracle   exact-rand   gen-rand   best generated labels"
    print(header)
    print("-" * len(header))
    for benchmark in benchmarks:
        hidden_labels = ", ".join(benchmark.target.slot_labels)
        best_labels = ", ".join(
            label if label is not None else "?"
            for label in benchmark.generated_random_report.candidates[0].slot_labels
        )
        print(
            f"{benchmark.target.name:<10} {hidden_labels:<50} "
            f"{benchmark.oracle_fidelity:>6.4f}   "
            f"{benchmark.exact_random_report.candidates[0].fidelity:>10.4f}   "
            f"{benchmark.generated_random_report.candidates[0].fidelity:>8.4f}   "
            f"{best_labels}"
        )


def summarize_hidden_shallow_circuit_benchmark(
    benchmarks: list[HiddenShallowCircuitBenchmark],
) -> list[HiddenShallowCircuitAggregate]:
    if not benchmarks:
        raise ValueError("summarize_hidden_shallow_circuit_benchmark needs at least one benchmark")

    return [
        _hidden_benchmark_aggregate("exact-grid", [item.exact_report for item in benchmarks]),
        _hidden_benchmark_aggregate(
            "generated-label-grid",
            [item.generated_label_grid_report for item in benchmarks],
        ),
        _hidden_benchmark_aggregate("generated-random", [item.random_report for item in benchmarks]),
    ]


def summarize_hidden_two_entangler_circuit_benchmark(
    benchmarks: list[HiddenTwoEntanglerCircuitBenchmark],
) -> list[HiddenShallowCircuitAggregate]:
    if not benchmarks:
        raise ValueError("summarize_hidden_two_entangler_circuit_benchmark needs at least one benchmark")

    return [
        HiddenShallowCircuitAggregate(
            mode="oracle",
            n_targets=len(benchmarks),
            mean_best=1.0,
            median_best=1.0,
            min_best=1.0,
            max_best=1.0,
            success_95=1.0,
            success_98=1.0,
            success_99=1.0,
        ),
        _hidden_benchmark_aggregate("exact-random", [item.exact_random_report for item in benchmarks]),
        _hidden_benchmark_aggregate("generated-random", [item.generated_random_report for item in benchmarks]),
    ]


def print_hidden_shallow_circuit_summary(
    benchmarks: list[HiddenShallowCircuitBenchmark] | list[HiddenShallowCircuitAggregate],
) -> None:
    aggregates = (
        benchmarks
        if benchmarks and isinstance(benchmarks[0], HiddenShallowCircuitAggregate)
        else summarize_hidden_shallow_circuit_benchmark(benchmarks)
    )
    header = "mode                   n   mean best   median   min      max      >=0.95   >=0.98   >=0.99"
    print(header)
    print("-" * len(header))
    for item in aggregates:
        print(
            f"{item.mode:<22} {item.n_targets:<3} "
            f"{item.mean_best:>9.4f}   {item.median_best:>6.4f}   "
            f"{item.min_best:>6.4f}   {item.max_best:>6.4f}   "
            f"{item.success_95:>6.1%}   {item.success_98:>6.1%}   {item.success_99:>6.1%}"
        )


def print_hidden_two_entangler_circuit_summary(
    benchmarks: list[HiddenTwoEntanglerCircuitBenchmark] | list[HiddenShallowCircuitAggregate],
) -> None:
    aggregates = (
        benchmarks
        if benchmarks and isinstance(benchmarks[0], HiddenShallowCircuitAggregate)
        else summarize_hidden_two_entangler_circuit_benchmark(benchmarks)
    )
    print_hidden_shallow_circuit_summary(aggregates)


def plot_hidden_shallow_circuit_best_fidelities(
    benchmarks: list[HiddenShallowCircuitBenchmark],
) -> None:
    if not benchmarks:
        raise ValueError("plot_hidden_shallow_circuit_best_fidelities needs at least one benchmark")

    values = [
        _hidden_benchmark_best_values([item.exact_report for item in benchmarks]),
        _hidden_benchmark_best_values([item.generated_label_grid_report for item in benchmarks]),
        _hidden_benchmark_best_values([item.random_report for item in benchmarks]),
    ]
    labels = ["exact grid", "generated label grid", "generated random"]

    plt.figure(figsize=(8, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Hidden shallow-circuit synthesis benchmark")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


def plot_hidden_two_entangler_best_fidelities(
    benchmarks: list[HiddenTwoEntanglerCircuitBenchmark],
) -> None:
    if not benchmarks:
        raise ValueError("plot_hidden_two_entangler_best_fidelities needs at least one benchmark")

    values = [
        torch.ones(len(benchmarks), dtype=torch.float32),
        _hidden_benchmark_best_values([item.exact_random_report for item in benchmarks]),
        _hidden_benchmark_best_values([item.generated_random_report for item in benchmarks]),
    ]
    labels = ["oracle", "exact random", "generated random"]

    plt.figure(figsize=(8, 4))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("best unitary fidelity")
    plt.title("Hidden two-entangler synthesis benchmark")
    plt.ylim(0.0, 1.02)
    plt.tight_layout()


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


def _batched_local_layer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nab,ncd->nacbd", a, b).reshape(a.shape[0], 4, 4)


def _first_index_for_label(local_labels: list[str], label: str) -> int:
    for i, local_label in enumerate(local_labels):
        if local_label == label:
            return i
    raise ValueError(f"No local gates found for label {label!r}")


def _labels_for_slots(slots: list[int], local_labels: list[str | None] | None) -> tuple[str | None, str | None, str | None, str | None]:
    if local_labels is None:
        return (None, None, None, None)
    return tuple(local_labels[i] for i in slots)


def _top_candidates_from_slots(
    fidelities: torch.Tensor,
    slot_indices: torch.Tensor,
    target: str,
    entangler: str,
    template: str,
    top_k: int,
    local_labels: list[str | None] | None,
) -> list[SynthesisCandidate]:
    values, rows = torch.topk(fidelities, k=min(top_k, fidelities.numel()))
    candidates = []
    for value, row in zip(values.tolist(), rows.tolist()):
        slots = slot_indices[row].tolist()
        candidates.append(
            SynthesisCandidate(
                target=target,
                template=template,
                entangler=entangler,
                fidelity=value,
                slot_indices=tuple(slots),
                slot_labels=_labels_for_slots(slots, local_labels),
            )
        )
    return candidates


def _hidden_benchmark_best_values(reports: list[SynthesisReport]) -> torch.Tensor:
    return torch.tensor([report.candidates[0].fidelity for report in reports], dtype=torch.float32)


def _hidden_benchmark_aggregate(mode: str, reports: list[SynthesisReport]) -> HiddenShallowCircuitAggregate:
    values = _hidden_benchmark_best_values(reports)
    return HiddenShallowCircuitAggregate(
        mode=mode,
        n_targets=values.numel(),
        mean_best=values.mean().item(),
        median_best=values.median().item(),
        min_best=values.min().item(),
        max_best=values.max().item(),
        success_95=(values >= 0.95).float().mean().item(),
        success_98=(values >= 0.98).float().mean().item(),
        success_99=(values >= 0.99).float().mean().item(),
    )
