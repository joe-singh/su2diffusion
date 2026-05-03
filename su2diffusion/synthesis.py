from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch


@dataclass(frozen=True)
class SynthesisCandidate:
    target: str
    template: str
    entangler: str
    fidelity: float
    slot_indices: tuple[int, int, int, int]
    slot_labels: tuple[str | None, str | None, str | None, str | None]


@dataclass(frozen=True)
class SynthesisReport:
    name: str
    mode: str
    target: str
    entangler: str
    candidates: list[SynthesisCandidate]
    fidelities: tuple[float, ...]


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


def unitary_fidelity_batch(candidates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dim = candidates.shape[-1]
    overlaps = torch.einsum("ij,nij->n", target.conj(), candidates).abs() / dim
    return overlaps.real


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
    return make_synthesis_report(
        candidates[:top_k],
        name=name or f"{target.upper()} guided search",
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
    units = quaternion_to_unitary(local_gates)
    target_unitary = two_qubit_gate(target, device=device)
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
        target=target,
        entangler=entangler,
        template="unconstrained-local-entangler-local",
        top_k=top_k,
        local_labels=local_labels,
    )
    return make_synthesis_report(
        candidates,
        name=name or f"{target.upper()} random generated search",
        mode="random",
        fidelities=fidelities.tolist(),
    )


def synthesize_named_gate_label_grid_report(
    local_gates: torch.Tensor,
    local_labels: list[str],
    target: str = "cz",
    entangler: str = "cz",
    top_k: int = 5,
    name: str | None = None,
) -> SynthesisReport:
    if local_gates.shape[0] == 0:
        raise ValueError("synthesize_named_gate_label_grid_report needs at least one local gate")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    unique_labels = list(dict.fromkeys(local_labels))
    representative_indices = [_first_index_for_label(local_labels, label) for label in unique_labels]
    representatives = local_gates[representative_indices]
    units = quaternion_to_unitary(representatives)

    target = target.lower()
    entangler = entangler.lower()
    device = local_gates.device
    target_unitary = two_qubit_gate(target, device=device)
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
                target=target,
                template="label-grid-local-entangler-local",
                entangler=entangler,
                fidelity=value,
                slot_indices=sample_slots,
                slot_labels=tuple(unique_labels[label_idx] for label_idx in label_slots),
            )
        )
    return make_synthesis_report(
        candidates,
        name=name or f"{target.upper()} label-grid search",
        mode="label-grid",
        fidelities=fidelities.tolist(),
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
