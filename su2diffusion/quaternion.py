import torch


def q_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def q_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions with components ordered as (w, x, y, z)."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)

    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw

    return torch.stack([w, x, y, z], dim=-1)


def q_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit quaternion."""
    w = q[..., :1]
    xyz = q[..., 1:]
    return torch.cat([w, -xyz], dim=-1)


def q_exp(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map R^3 -> SU(2)."""
    theta = v.norm(dim=-1, keepdim=True)
    direction = v / theta.clamp_min(eps)

    w = torch.cos(theta)
    xyz = torch.sin(theta) * direction

    small = theta < 1e-6
    xyz = torch.where(small, v, xyz)

    return q_normalize(torch.cat([w, xyz], dim=-1))


def q_log(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Principal logarithm SU(2) -> R^3."""
    q = q_normalize(q)

    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]

    sin_theta = xyz.norm(dim=-1, keepdim=True)
    theta = torch.atan2(sin_theta, w)

    direction = xyz / sin_theta.clamp_min(eps)
    v = theta * direction

    small = sin_theta < 1e-6
    return torch.where(small, xyz, v)


def sample_haar(
    n: int,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Haar-random SU(2) by normalizing a Gaussian in R^4."""
    q = torch.randn(n, 4, device=device, generator=generator)
    return q_normalize(q)


def su2_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Principal geodesic-like distance using ||log(q1^{-1} q2)||."""
    rel = q_mul(q_inv(q1), q2)
    return q_log(rel).norm(dim=-1)
