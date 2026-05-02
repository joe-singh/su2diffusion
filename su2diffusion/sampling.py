import torch

from .diffusion import DiffusionSchedule
from .quaternion import q_exp, q_mul, q_normalize, sample_haar


@torch.no_grad()
def sample_reverse(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    n_samples: int = 5000,
    eta: float = 1.0,
    labels: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """DDPM-style reverse sampler on SU(2)."""
    device = device or next(model.parameters()).device
    if labels is not None:
        labels = labels.to(device=device, dtype=torch.long)
        if labels.shape != (n_samples,):
            raise ValueError(f"Expected labels with shape ({n_samples},), got {tuple(labels.shape)}")

    betas, _, sigmas = schedule.tensors(device)
    q = sample_haar(n_samples, device=device)

    for s in reversed(range(schedule.T)):
        t_idx = torch.full((n_samples,), s + 1, device=device, dtype=torch.long)
        eps_pred = model(q, t_idx, labels=labels)

        beta = betas[s]
        sigma = sigmas[s]
        drift = -(beta / sigma.clamp_min(1e-8)) * eps_pred

        if s > 0 and eta > 0:
            z = torch.randn(n_samples, 3, device=device)
            noise = eta * torch.sqrt(beta) * z
        else:
            noise = torch.zeros_like(drift)

        q = q_mul(q, q_exp(drift + noise))
        q = q_normalize(q)

    return q


@torch.no_grad()
def sample_reverse_trajectory(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    n_samples: int = 3000,
    eta: float = 1.0,
    record_every: int = 2,
    labels: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    """Run reverse sampling and record intermediate samples."""
    device = device or next(model.parameters()).device
    if labels is not None:
        labels = labels.to(device=device, dtype=torch.long)
        if labels.shape != (n_samples,):
            raise ValueError(f"Expected labels with shape ({n_samples},), got {tuple(labels.shape)}")

    betas, _, sigmas = schedule.tensors(device)
    q = sample_haar(n_samples, device=device)

    frames = [q.detach().cpu().clone()]
    t_values = [schedule.T]

    for s in reversed(range(schedule.T)):
        t_idx = torch.full((n_samples,), s + 1, device=device, dtype=torch.long)
        eps_pred = model(q, t_idx, labels=labels)

        beta = betas[s]
        sigma = sigmas[s]
        drift = -(beta / sigma.clamp_min(1e-8)) * eps_pred

        if s > 0 and eta > 0:
            z = torch.randn(n_samples, 3, device=device)
            noise = eta * torch.sqrt(beta) * z
        else:
            noise = torch.zeros_like(drift)

        q = q_mul(q, q_exp(drift + noise))
        q = q_normalize(q)

        if (s % record_every == 0) or (s == 0):
            frames.append(q.detach().cpu().clone())
            t_values.append(s)

    return q, frames, t_values
