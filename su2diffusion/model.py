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
    def __init__(
        self,
        T: int = 200,
        time_dim: int = 64,
        hidden: int = 512,
        num_labels: int | None = None,
        label_dim: int = 32,
    ):
        super().__init__()
        self.T = T
        self.time_dim = time_dim
        self.num_labels = num_labels
        self.label_dim = label_dim if num_labels is not None else 0

        if num_labels is None:
            self.label_embedding = None
        else:
            self.label_embedding = nn.Embedding(num_labels, label_dim)

        self.net = nn.Sequential(
            nn.Linear(4 + time_dim + self.label_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, q: torch.Tensor, t_idx: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        t_scaled = t_idx.float() / self.T
        temb = timestep_embedding(t_scaled, self.time_dim)
        parts = [q, temb]

        if self.label_embedding is not None:
            if labels is None:
                raise ValueError("Conditional SU2Denoiser requires labels")
            parts.append(self.label_embedding(labels))
        elif labels is not None:
            raise ValueError("Unconditional SU2Denoiser does not accept labels")

        x = torch.cat(parts, dim=-1)
        return self.net(x)


class CircuitDenoiser(nn.Module):
    def __init__(
        self,
        T: int = 200,
        n_slots: int = 6,
        time_dim: int = 64,
        hidden: int = 512,
    ):
        super().__init__()
        self.T = T
        self.n_slots = n_slots
        self.time_dim = time_dim

        self.net = nn.Sequential(
            nn.Linear(n_slots * 4 + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_slots * 3),
        )

    def forward(self, q_stack: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        if q_stack.ndim != 3 or q_stack.shape[1:] != (self.n_slots, 4):
            raise ValueError(f"Expected q_stack with shape (batch, {self.n_slots}, 4)")

        t_scaled = t_idx.float() / self.T
        temb = timestep_embedding(t_scaled, self.time_dim)
        x = torch.cat([q_stack.reshape(q_stack.shape[0], self.n_slots * 4), temb], dim=-1)
        return self.net(x).reshape(q_stack.shape[0], self.n_slots, 3)


class TargetConditionedCircuitDenoiser(nn.Module):
    def __init__(
        self,
        T: int = 200,
        n_slots: int = 6,
        target_dim: int = 32,
        time_dim: int = 64,
        hidden: int = 512,
    ):
        super().__init__()
        self.T = T
        self.n_slots = n_slots
        self.target_dim = target_dim
        self.time_dim = time_dim

        self.net = nn.Sequential(
            nn.Linear(n_slots * 4 + target_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_slots * 3),
        )

    def forward(self, q_stack: torch.Tensor, t_idx: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        if q_stack.ndim != 3 or q_stack.shape[1:] != (self.n_slots, 4):
            raise ValueError(f"Expected q_stack with shape (batch, {self.n_slots}, 4)")
        if target_features.ndim != 2 or target_features.shape != (q_stack.shape[0], self.target_dim):
            raise ValueError(f"Expected target_features with shape (batch, {self.target_dim})")

        t_scaled = t_idx.float() / self.T
        temb = timestep_embedding(t_scaled, self.time_dim)
        x = torch.cat(
            [q_stack.reshape(q_stack.shape[0], self.n_slots * 4), target_features, temb],
            dim=-1,
        )
        return self.net(x).reshape(q_stack.shape[0], self.n_slots, 3)


class TargetLabelConditionedCircuitDenoiser(nn.Module):
    def __init__(
        self,
        T: int = 200,
        n_slots: int = 6,
        target_dim: int = 32,
        num_labels: int = 24,
        label_dim: int = 16,
        time_dim: int = 64,
        hidden: int = 512,
    ):
        super().__init__()
        self.T = T
        self.n_slots = n_slots
        self.target_dim = target_dim
        self.num_labels = num_labels
        self.label_dim = label_dim
        self.time_dim = time_dim
        self.label_embedding = nn.Embedding(num_labels, label_dim)

        self.net = nn.Sequential(
            nn.Linear(n_slots * 4 + target_dim + n_slots * label_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_slots * 3),
        )

    def forward(
        self,
        q_stack: torch.Tensor,
        t_idx: torch.Tensor,
        target_features: torch.Tensor,
        slot_labels: torch.Tensor,
    ) -> torch.Tensor:
        if q_stack.ndim != 3 or q_stack.shape[1:] != (self.n_slots, 4):
            raise ValueError(f"Expected q_stack with shape (batch, {self.n_slots}, 4)")
        if target_features.ndim != 2 or target_features.shape != (q_stack.shape[0], self.target_dim):
            raise ValueError(f"Expected target_features with shape (batch, {self.target_dim})")
        if slot_labels.ndim != 2 or slot_labels.shape != (q_stack.shape[0], self.n_slots):
            raise ValueError(f"Expected slot_labels with shape (batch, {self.n_slots})")

        t_scaled = t_idx.float() / self.T
        temb = timestep_embedding(t_scaled, self.time_dim)
        labels = self.label_embedding(slot_labels).reshape(q_stack.shape[0], self.n_slots * self.label_dim)
        x = torch.cat(
            [q_stack.reshape(q_stack.shape[0], self.n_slots * 4), target_features, labels, temb],
            dim=-1,
        )
        return self.net(x).reshape(q_stack.shape[0], self.n_slots, 3)
