from dataclasses import dataclass

import torch

from .quaternion import q_exp, q_mul, q_normalize


@dataclass(frozen=True)
class BlobConfig:
    sigma_data: float = 0.18


@dataclass(frozen=True)
class DataConfig:
    kind: str = "blobs"
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


def gate_centers(device: torch.device | str | None = None) -> torch.Tensor:
    inv_sqrt2 = 2.0**-0.5
    centers = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # I
            [0.0, 1.0, 0.0, 0.0],  # X up to global phase
            [0.0, 0.0, 1.0, 0.0],  # Y up to global phase
            [0.0, 0.0, 0.0, 1.0],  # Z up to global phase
            [inv_sqrt2, inv_sqrt2, 0.0, 0.0],  # sqrt(X)
            [inv_sqrt2, 0.0, inv_sqrt2, 0.0],  # sqrt(Y)
            [inv_sqrt2, 0.0, 0.0, inv_sqrt2],  # sqrt(Z)
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


def centers_for_config(config: DataConfig | BlobConfig | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    if config is None or isinstance(config, BlobConfig) or config.kind == "blobs":
        return default_centers(device=device)
    if config.kind == "gates":
        return gate_centers(device=device)
    raise ValueError(f"Unknown data kind {config.kind!r}")


def sample_clean(
    batch_size: int,
    centers: torch.Tensor | None = None,
    config: DataConfig | BlobConfig | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config is None:
        config = DataConfig()
    if isinstance(config, BlobConfig):
        data_config = DataConfig(kind="blobs", sigma_data=config.sigma_data)
    else:
        data_config = config

    centers = centers if centers is not None else centers_for_config(data_config, device=device)
    return sample_clean_blobs(
        batch_size=batch_size,
        centers=centers,
        config=BlobConfig(sigma_data=data_config.sigma_data),
    )
