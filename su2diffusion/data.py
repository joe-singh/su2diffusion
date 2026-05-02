from dataclasses import dataclass

import torch

from .quaternion import q_exp, q_mul, q_normalize


@dataclass(frozen=True)
class BlobConfig:
    sigma_data: float = 0.18


def default_centers(device: torch.device | str | None = None) -> torch.Tensor:
    centers = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    return q_normalize(centers)


def sample_clean_blobs(
    batch_size: int,
    centers: torch.Tensor | None = None,
    config: BlobConfig | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample clean data U_0 from a mixture of local SU(2) blobs."""
    config = config or BlobConfig()
    centers = centers if centers is not None else default_centers(device=device)
    device = centers.device

    labels = torch.randint(0, centers.shape[0], (batch_size,), device=device)
    c = centers[labels]

    noise = config.sigma_data * torch.randn(batch_size, 3, device=device)
    u = q_mul(c, q_exp(noise))

    return q_normalize(u), labels
