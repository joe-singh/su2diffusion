from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import BlobConfig, DataConfig, centers_for_config, sample_clean
from .device import get_default_device
from .diffusion import DiffusionSchedule, brownian_forward_heat_target
from .model import SU2Denoiser


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 2048
    num_steps: int = 2000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    hidden: int = 512
    n_terms: int = 128
    seed: int = 0
    conditional: bool = False
    label_dim: int = 32


def train_heat_kernel_model(
    train_config: TrainConfig | None = None,
    schedule: DiffusionSchedule | None = None,
    blob_config: BlobConfig | DataConfig | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> tuple[SU2Denoiser, list[float]]:
    train_config = train_config or TrainConfig()
    schedule = schedule or DiffusionSchedule()
    blob_config = blob_config or BlobConfig()
    device = torch.device(device) if device is not None else get_default_device()

    torch.manual_seed(train_config.seed)

    centers = centers_for_config(blob_config, device=device)
    num_labels = centers.shape[0] if train_config.conditional else None
    model = SU2Denoiser(
        T=schedule.T,
        hidden=train_config.hidden,
        num_labels=num_labels,
        label_dim=train_config.label_dim,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    losses: list[float] = []

    iterator = range(1, train_config.num_steps + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Training heat-kernel target", dynamic_ncols=True)

    for step in iterator:
        q0, labels = sample_clean(
            train_config.batch_size,
            centers=centers,
            config=blob_config,
        )
        t_idx = torch.randint(1, schedule.T + 1, (train_config.batch_size,), device=device)

        with torch.no_grad():
            qt, eps_target = brownian_forward_heat_target(
                q0,
                t_idx,
                schedule=schedule,
                n_terms=train_config.n_terms,
            )

        eps_pred = model(qt, t_idx, labels=labels if train_config.conditional else None)
        loss = F.mse_loss(eps_pred, eps_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_value = loss.item()
        losses.append(loss_value)

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_value:.5f}"})

    return model, losses
