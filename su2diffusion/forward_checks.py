from dataclasses import dataclass

import torch

from .data import BlobConfig, default_centers, sample_clean_blobs
from .diagnostics import wasserstein_1d
from .diffusion import DiffusionSchedule, brownian_forward_heat_target
from .quaternion import sample_haar
from .viz import nearest_center_dist


@dataclass(frozen=True)
class ForwardProcessDiagnostics:
    timesteps: list[int]
    mean_nearest_center_distance: list[float]
    norm_error_mean: float
    final_distance_to_haar_w1: float
    target_abs_mean: float
    target_abs_max: float
    target_finite: bool


@torch.no_grad()
def diagnose_forward_process(
    schedule: DiffusionSchedule | None = None,
    blob_config: BlobConfig | None = None,
    batch_size: int = 1024,
    timesteps: list[int] | None = None,
    n_terms: int = 64,
    device: torch.device | str = "cpu",
) -> ForwardProcessDiagnostics:
    schedule = schedule or DiffusionSchedule()
    blob_config = blob_config or BlobConfig()
    device = torch.device(device)
    centers = default_centers(device=device)

    if timesteps is None:
        timesteps = [1, max(1, schedule.T // 4), max(1, schedule.T // 2), schedule.T]
    timesteps = sorted(set(max(1, min(schedule.T, int(t))) for t in timesteps))

    q0, _ = sample_clean_blobs(batch_size, centers=centers, config=blob_config)

    mean_distances = []
    norm_errors = []
    final_qt = None
    final_target = None

    for timestep in timesteps:
        t_idx = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
        qt, eps_target = brownian_forward_heat_target(
            q0,
            t_idx,
            schedule=schedule,
            n_terms=n_terms,
        )
        mean_distances.append(nearest_center_dist(qt, centers=centers).mean().item())
        norm_errors.append((qt.norm(dim=-1) - 1.0).abs().mean().item())
        final_qt = qt
        final_target = eps_target

    assert final_qt is not None
    assert final_target is not None

    q_haar = sample_haar(batch_size, device=device)
    final_dist = nearest_center_dist(final_qt, centers=centers)
    haar_dist = nearest_center_dist(q_haar, centers=centers)

    return ForwardProcessDiagnostics(
        timesteps=timesteps,
        mean_nearest_center_distance=mean_distances,
        norm_error_mean=max(norm_errors),
        final_distance_to_haar_w1=wasserstein_1d(final_dist, haar_dist),
        target_abs_mean=final_target.abs().mean().item(),
        target_abs_max=final_target.abs().max().item(),
        target_finite=torch.isfinite(final_target).all().item(),
    )


def print_forward_diagnostics(diagnostics: ForwardProcessDiagnostics) -> None:
    print("forward process diagnostics")
    print("---------------------------")
    print("timestep distance means:")
    for timestep, distance in zip(diagnostics.timesteps, diagnostics.mean_nearest_center_distance):
        print(f"  t={timestep:<4} mean nearest-center distance={distance:.4f}")
    print(f"max norm error mean:        {diagnostics.norm_error_mean:.2e}")
    print(f"final W1 to Haar distance:  {diagnostics.final_distance_to_haar_w1:.4f}")
    print(f"target abs mean/max:        {diagnostics.target_abs_mean:.4f} / {diagnostics.target_abs_max:.4f}")
    print(f"target finite:              {diagnostics.target_finite}")
