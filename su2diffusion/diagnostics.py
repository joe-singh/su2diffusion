from dataclasses import dataclass

import torch

from .data import default_centers
from .viz import nearest_center_dist


@dataclass(frozen=True)
class DistanceSummary:
    mean: float
    std: float
    q05: float
    q50: float
    q95: float


@dataclass(frozen=True)
class SampleDiagnostics:
    norm_error_mean: float
    distance_to_clean_w1: float
    distance_to_haar_w1: float
    nearest_center_distance: DistanceSummary
    nearest_center_mass: list[float]
    clean_center_mass: list[float]
    center_mass_l1: float
    per_center_distance: list[DistanceSummary] | None = None
    projective_nearest_center_distance: DistanceSummary | None = None
    projective_distance_to_clean_w1: float | None = None


@dataclass(frozen=True)
class ConditionalLabelDiagnostics:
    accuracy: float
    per_label_accuracy: list[float]
    requested_center_distance: list[DistanceSummary]


def wasserstein_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """Empirical 1-Wasserstein distance between two 1D samples."""
    x = x.detach().flatten().float().cpu().sort().values
    y = y.detach().flatten().float().cpu().sort().values
    n = min(x.numel(), y.numel())
    if n == 0:
        raise ValueError("wasserstein_1d needs non-empty samples")
    return (x[:n] - y[:n]).abs().mean().item()


def summarize_distances(distances: torch.Tensor) -> DistanceSummary:
    distances = distances.detach().flatten().float().cpu()
    quantiles = torch.quantile(distances, torch.tensor([0.05, 0.5, 0.95]))
    return DistanceSummary(
        mean=distances.mean().item(),
        std=distances.std(unbiased=False).item(),
        q05=quantiles[0].item(),
        q50=quantiles[1].item(),
        q95=quantiles[2].item(),
    )


@torch.no_grad()
def nearest_center_labels(
    q: torch.Tensor,
    centers: torch.Tensor | None = None,
    projective: bool = False,
) -> torch.Tensor:
    centers = centers if centers is not None else default_centers(device=q.device)
    return _center_distance_matrix(q, centers=centers, projective=projective).argmin(dim=1)


@torch.no_grad()
def nearest_center_mass(
    q: torch.Tensor,
    centers: torch.Tensor | None = None,
    projective: bool = False,
) -> list[float]:
    centers = centers if centers is not None else default_centers(device=q.device)
    labels = nearest_center_labels(q, centers=centers, projective=projective)
    return _mass_from_labels(labels, centers.shape[0])


def _mass_from_labels(labels: torch.Tensor, n_centers: int) -> list[float]:
    counts = torch.bincount(labels.cpu(), minlength=n_centers).float()
    return (counts / counts.sum().clamp_min(1.0)).tolist()


@torch.no_grad()
def _center_distance_matrix(
    q: torch.Tensor,
    centers: torch.Tensor,
    projective: bool = False,
) -> torch.Tensor:
    dists = []
    for k in range(centers.shape[0]):
        c = centers[k].view(1, 4).expand(q.shape[0], 4)
        if projective:
            d = torch.minimum(nearest_center_dist(q, centers=c), nearest_center_dist(q, centers=-c))
        else:
            d = nearest_center_dist(q, centers=c)
        dists.append(d)
    return torch.stack(dists, dim=1)


@torch.no_grad()
def per_center_distance_summary(
    q: torch.Tensor,
    centers: torch.Tensor | None = None,
    projective: bool = False,
) -> list[DistanceSummary]:
    centers = centers if centers is not None else default_centers(device=q.device)
    dists = _center_distance_matrix(q, centers=centers, projective=projective)
    labels = dists.argmin(dim=1)
    return _per_center_distance_summary_from_matrix(dists, labels)


def _per_center_distance_summary_from_matrix(
    dists: torch.Tensor,
    labels: torch.Tensor,
) -> list[DistanceSummary]:
    summaries = []
    for k in range(dists.shape[1]):
        assigned = dists[labels == k, k]
        if assigned.numel() == 0:
            assigned = torch.tensor([float("nan")], device=dists.device)
        summaries.append(summarize_distances(assigned))
    return summaries


@torch.no_grad()
def projective_nearest_center_dist(q: torch.Tensor, centers: torch.Tensor | None = None) -> torch.Tensor:
    centers = centers if centers is not None else default_centers(device=q.device)
    return _center_distance_matrix(q, centers=centers, projective=True).min(dim=1).values


@torch.no_grad()
def diagnose_samples(
    generated: torch.Tensor,
    clean_reference: torch.Tensor,
    haar_reference: torch.Tensor,
    centers: torch.Tensor | None = None,
    include_per_center: bool = False,
    include_projective: bool = False,
) -> SampleDiagnostics:
    """Compare generated samples to clean data and Haar baselines."""
    centers = centers if centers is not None else default_centers(device=generated.device)
    clean_reference = clean_reference.to(generated.device)
    haar_reference = haar_reference.to(generated.device)

    generated_matrix = _center_distance_matrix(generated, centers=centers)
    clean_matrix = _center_distance_matrix(clean_reference, centers=centers)
    haar_matrix = _center_distance_matrix(haar_reference, centers=centers)

    d_generated, generated_labels = generated_matrix.min(dim=1)
    d_clean, clean_labels = clean_matrix.min(dim=1)
    d_haar = haar_matrix.min(dim=1).values

    generated_mass = _mass_from_labels(generated_labels, centers.shape[0])
    clean_mass = _mass_from_labels(clean_labels, centers.shape[0])
    mass_l1 = sum(abs(a - b) for a, b in zip(generated_mass, clean_mass))

    per_center = None
    if include_per_center:
        per_center = _per_center_distance_summary_from_matrix(generated_matrix, generated_labels)

    projective_summary = None
    projective_w1 = None
    if include_projective:
        pd_generated = _center_distance_matrix(generated, centers=centers, projective=True).min(dim=1).values
        pd_clean = _center_distance_matrix(clean_reference, centers=centers, projective=True).min(dim=1).values
        projective_summary = summarize_distances(pd_generated)
        projective_w1 = wasserstein_1d(pd_generated, pd_clean)

    return SampleDiagnostics(
        norm_error_mean=(generated.norm(dim=-1) - 1.0).abs().mean().item(),
        distance_to_clean_w1=wasserstein_1d(d_generated, d_clean),
        distance_to_haar_w1=wasserstein_1d(d_generated, d_haar),
        nearest_center_distance=summarize_distances(d_generated),
        nearest_center_mass=generated_mass,
        clean_center_mass=clean_mass,
        center_mass_l1=mass_l1,
        per_center_distance=per_center,
        projective_nearest_center_distance=projective_summary,
        projective_distance_to_clean_w1=projective_w1,
    )


@torch.no_grad()
def diagnose_conditional_labels(
    generated: torch.Tensor,
    requested_labels: torch.Tensor,
    centers: torch.Tensor | None = None,
) -> ConditionalLabelDiagnostics:
    centers = centers if centers is not None else default_centers(device=generated.device)
    requested_labels = requested_labels.to(device=generated.device, dtype=torch.long)
    if requested_labels.shape != (generated.shape[0],):
        raise ValueError(f"Expected requested_labels with shape ({generated.shape[0]},), got {tuple(requested_labels.shape)}")

    dists = _center_distance_matrix(generated, centers=centers)
    predicted_labels = dists.argmin(dim=1)
    correct = predicted_labels == requested_labels
    requested_distances = dists.gather(1, requested_labels.view(-1, 1)).squeeze(1)

    per_label_accuracy = []
    per_label_distance = []
    for label in range(centers.shape[0]):
        mask = requested_labels == label
        if mask.any():
            per_label_accuracy.append(correct[mask].float().mean().item())
            per_label_distance.append(summarize_distances(requested_distances[mask]))
        else:
            per_label_accuracy.append(float("nan"))
            per_label_distance.append(summarize_distances(torch.tensor([float("nan")], device=generated.device)))

    return ConditionalLabelDiagnostics(
        accuracy=correct.float().mean().item(),
        per_label_accuracy=per_label_accuracy,
        requested_center_distance=per_label_distance,
    )


def print_diagnostics_table(results: dict[str, SampleDiagnostics]) -> None:
    header = "sample                 W1 clean   W1 Haar   mass L1   dist mean   dist q50   norm err"
    print(header)
    print("-" * len(header))
    for name, diag in results.items():
        print(
            f"{name:<20} "
            f"{diag.distance_to_clean_w1:>8.4f} "
            f"{diag.distance_to_haar_w1:>9.4f} "
            f"{diag.center_mass_l1:>9.4f} "
            f"{diag.nearest_center_distance.mean:>10.4f} "
            f"{diag.nearest_center_distance.q50:>9.4f} "
            f"{diag.norm_error_mean:>10.2e}"
        )


def _format_center_names(n_centers: int, center_names: list[str] | None = None) -> list[str]:
    if center_names is None:
        return [str(i) for i in range(n_centers)]
    if len(center_names) != n_centers:
        raise ValueError(f"Expected {n_centers} center names, got {len(center_names)}")
    return center_names


def print_center_mass_table(results: dict[str, SampleDiagnostics], center_names: list[str] | None = None) -> None:
    if not results:
        raise ValueError("print_center_mass_table needs at least one diagnostics entry")

    first = next(iter(results.values()))
    names = _format_center_names(len(first.clean_center_mass), center_names=center_names)
    sample_names = list(results)
    header = f"{'center':<10} {'clean':>8}"
    for sample_name in sample_names:
        header += f" {sample_name[:13]:>13} {'delta':>9}"

    print(header)
    print("-" * len(header))
    for i, center_name in enumerate(names):
        clean_mass = first.clean_center_mass[i]
        row = f"{center_name:<10} {clean_mass:>8.4f}"
        for sample_name in sample_names:
            generated_mass = results[sample_name].nearest_center_mass[i]
            row += f" {generated_mass:>13.4f} {generated_mass - clean_mass:>+9.4f}"
        print(row)


def print_per_center_table(results: dict[str, SampleDiagnostics], center_names: list[str] | None = None) -> None:
    for name, diag in results.items():
        if diag.per_center_distance is None:
            print(f"{name} per-center nearest-distance mean: not computed")
            continue
        names = _format_center_names(len(diag.per_center_distance), center_names=center_names)
        print(f"{name} per-center nearest-distance mean:")
        print("  " + " ".join(f"{center}:{summary.mean:.4f}" for center, summary in zip(names, diag.per_center_distance)))


def print_conditional_label_table(
    results: dict[str, ConditionalLabelDiagnostics],
    center_names: list[str] | None = None,
) -> None:
    if not results:
        raise ValueError("print_conditional_label_table needs at least one diagnostics entry")

    first = next(iter(results.values()))
    names = _format_center_names(len(first.per_label_accuracy), center_names=center_names)
    header = f"{'requested':<10}"
    for sample_name in results:
        header += f" {sample_name[:13]:>13} {'dist mean':>10}"

    print(header)
    print("-" * len(header))
    for i, center_name in enumerate(names):
        row = f"{center_name:<10}"
        for diag in results.values():
            row += f" {diag.per_label_accuracy[i]:>13.4f} {diag.requested_center_distance[i].mean:>10.4f}"
        print(row)

    print()
    summary = "overall   "
    for diag in results.values():
        summary += f" {diag.accuracy:>13.4f} {'':>10}"
    print(summary.rstrip())
