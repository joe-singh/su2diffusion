import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=t.device).float()
        / max(half - 1, 1)
    )

    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


class SU2Denoiser(nn.Module):
    def __init__(self, T: int = 200, time_dim: int = 64, hidden: int = 512):
        super().__init__()
        self.T = T
        self.time_dim = time_dim

        self.net = nn.Sequential(
            nn.Linear(4 + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, q: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        t_scaled = t_idx.float() / self.T
        temb = timestep_embedding(t_scaled, self.time_dim)
        x = torch.cat([q, temb], dim=-1)
        return self.net(x)
