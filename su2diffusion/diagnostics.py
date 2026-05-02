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
def nearest_center_labels(q: torch.Tensor, centers: torch.Tensor | None = None) -> torch.Tensor:
    centers = centers if centers is not None else default_centers(device=q.device)
    dists = []
    for k in range(centers.shape[0]):
        c = centers[k].view(1, 4).expand(q.shape[0], 4)
        dists.append(nearest_center_dist(q, centers=c))
    return torch.stack(dists, dim=1).argmin(dim=1)


@torch.no_grad()
def nearest_center_mass(q: torch.Tensor, centers: torch.Tensor | None = None) -> list[float]:
    centers = centers if centers is not None else default_centers(device=q.device)
    labels = nearest_center_labels(q, centers=centers)
    counts = torch.bincount(labels.cpu(), minlength=centers.shape[0]).float()
    return (counts / counts.sum().clamp_min(1.0)).tolist()


@torch.no_grad()
def diagnose_samples(
    generated: torch.Tensor,
    clean_reference: torch.Tensor,
    haar_reference: torch.Tensor,
    centers: torch.Tensor | None = None,
) -> SampleDiagnostics:
    """Compare generated samples to clean data and Haar baselines."""
    centers = centers if centers is not None else default_centers(device=generated.device)

    d_generated = nearest_center_dist(generated, centers=centers)
    d_clean = nearest_center_dist(clean_reference.to(generated.device), centers=centers)
    d_haar = nearest_center_dist(haar_reference.to(generated.device), centers=centers)

    return SampleDiagnostics(
        norm_error_mean=(generated.norm(dim=-1) - 1.0).abs().mean().item(),
        distance_to_clean_w1=wasserstein_1d(d_generated, d_clean),
        distance_to_haar_w1=wasserstein_1d(d_generated, d_haar),
        nearest_center_distance=summarize_distances(d_generated),
        nearest_center_mass=nearest_center_mass(generated, centers=centers),
    )


def print_diagnostics_table(results: dict[str, SampleDiagnostics]) -> None:
    header = "sample                 W1 clean   W1 Haar   dist mean   dist q50   norm err"
    print(header)
    print("-" * len(header))
    for name, diag in results.items():
        print(
            f"{name:<20} "
            f"{diag.distance_to_clean_w1:>8.4f} "
            f"{diag.distance_to_haar_w1:>9.4f} "
            f"{diag.nearest_center_distance.mean:>10.4f} "
            f"{diag.nearest_center_distance.q50:>9.4f} "
            f"{diag.norm_error_mean:>10.2e}"
        )
