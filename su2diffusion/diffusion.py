import math
from dataclasses import dataclass

import torch

from .quaternion import q_exp, q_inv, q_log, q_mul, q_normalize


@dataclass(frozen=True)
class DiffusionSchedule:
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.005

    def tensors(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = torch.linspace(self.beta_start, self.beta_end, self.T, device=device)
        sigma2 = torch.cumsum(betas, dim=0)
        sigmas = torch.sqrt(sigma2)
        return betas, sigma2, sigmas


def su2_heat_kernel_radial_score(
    theta: torch.Tensor,
    sigma2_t: torch.Tensor,
    n_terms: int = 128,
    local_sigma_cutoff: float = 0.20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute d/dtheta log K_t(theta) for SU(2) ~= S^3."""
    orig_dtype = theta.dtype

    theta = theta.double()
    sigma2_t = sigma2_t.double()

    theta_safe = theta.clamp(min=1e-7, max=math.pi - 1e-7)

    sin_th = torch.sin(theta_safe)
    cos_th = torch.cos(theta_safe)
    local_radial = 1.0 / theta_safe - cos_th / sin_th - theta_safe / sigma2_t.clamp_min(eps)

    small_theta = theta < 1e-4
    local_series = (
        theta / 3.0
        + theta**3 / 45.0
        + 2.0 * theta**5 / 945.0
        - theta / sigma2_t.clamp_min(eps)
    )
    local_radial = torch.where(small_theta, local_series, local_radial)

    m = torch.arange(1, n_terms + 1, device=theta.device, dtype=torch.float64).view(1, -1)
    th = theta_safe.view(-1, 1)
    s2 = sigma2_t.view(-1, 1)

    exp_factor = torch.exp(-0.5 * (m**2 - 1.0) * s2)
    sin_mth = torch.sin(m * th)
    cos_mth = torch.cos(m * th)
    sin_th_b = torch.sin(th)
    cos_th_b = torch.cos(th)

    ratio = sin_mth / sin_th_b.clamp_min(eps)
    terms = m * exp_factor * ratio
    k_val = terms.sum(dim=1, keepdim=True)

    d_ratio = (m * cos_mth * sin_th_b - sin_mth * cos_th_b) / (sin_th_b**2).clamp_min(eps)
    d_terms = m * exp_factor * d_ratio
    d_k = d_terms.sum(dim=1, keepdim=True)

    spectral_radial = d_k / k_val.clamp_min(eps)
    spectral_ok = torch.isfinite(spectral_radial) & torch.isfinite(k_val) & (k_val > eps)
    use_local = sigma2_t.sqrt() < local_sigma_cutoff

    radial = torch.where(use_local | (~spectral_ok), local_radial, spectral_radial)
    return radial.to(orig_dtype)


def heat_epsilon_target(
    q0: torch.Tensor,
    qt: torch.Tensor,
    t_idx: torch.Tensor,
    schedule: DiffusionSchedule,
    n_terms: int = 128,
) -> torch.Tensor:
    """Heat-kernel-corrected epsilon target."""
    _, sigma2, sigmas = schedule.tensors(q0.device)
    batch_size = q0.shape[0]

    rel = q_mul(q_inv(q0), qt)
    xi = q_log(rel)

    theta = xi.norm(dim=-1, keepdim=True)
    n_hat = xi / theta.clamp_min(1e-8)

    sigma_t = sigmas[t_idx - 1].view(batch_size, 1)
    sigma2_t = sigma2[t_idx - 1].view(batch_size, 1)

    radial_score = su2_heat_kernel_radial_score(theta, sigma2_t, n_terms=n_terms)
    return -sigma_t * radial_score * n_hat


@torch.no_grad()
def brownian_forward_heat_target(
    q0: torch.Tensor,
    t_idx: torch.Tensor,
    schedule: DiffusionSchedule,
    n_terms: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Brownian-product forward noising and return the heat-kernel target."""
    betas, _, _ = schedule.tensors(q0.device)
    qt = q0.clone()

    for s in range(schedule.T):
        active = t_idx > s
        if active.any():
            n_active = active.sum().item()
            x = torch.randn(n_active, 3, device=q0.device)
            inc = q_exp(torch.sqrt(betas[s]) * x)
            qt[active] = q_mul(qt[active], inc)

    qt = q_normalize(qt)
    eps_target = heat_epsilon_target(q0, qt, t_idx, schedule=schedule, n_terms=n_terms)

    return qt, eps_target
