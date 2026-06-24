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


class SlotwiseTargetConditionedCircuitDenoiser(nn.Module):
    def __init__(
        self,
        T: int = 200,
        n_slots: int = 6,
        target_dim: int = 32,
        time_dim: int = 64,
        slot_dim: int = 16,
        hidden: int = 512,
    ):
        super().__init__()
        self.T = T
        self.n_slots = n_slots
        self.target_dim = target_dim
        self.time_dim = time_dim
        self.slot_dim = slot_dim
        self.slot_embedding = nn.Embedding(n_slots, slot_dim)

        self.net = nn.Sequential(
            nn.Linear(4 + target_dim + time_dim + slot_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, q_stack: torch.Tensor, t_idx: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        if q_stack.ndim != 3 or q_stack.shape[1:] != (self.n_slots, 4):
            raise ValueError(f"Expected q_stack with shape (batch, {self.n_slots}, 4)")
        if target_features.ndim != 2 or target_features.shape != (q_stack.shape[0], self.target_dim):
            raise ValueError(f"Expected target_features with shape (batch, {self.target_dim})")

        batch = q_stack.shape[0]
        t_scaled = t_idx.float() / self.T
        temb = timestep_embedding(t_scaled, self.time_dim)
        slot_ids = torch.arange(self.n_slots, device=q_stack.device)
        slot_emb = self.slot_embedding(slot_ids)

        x = torch.cat(
            [
                q_stack.reshape(batch * self.n_slots, 4),
                target_features[:, None, :].expand(batch, self.n_slots, self.target_dim).reshape(
                    batch * self.n_slots,
                    self.target_dim,
                ),
                temb[:, None, :].expand(batch, self.n_slots, self.time_dim).reshape(
                    batch * self.n_slots,
                    self.time_dim,
                ),
                slot_emb[None, :, :].expand(batch, self.n_slots, self.slot_dim).reshape(
                    batch * self.n_slots,
                    self.slot_dim,
                ),
            ],
            dim=-1,
        )
        return self.net(x).reshape(batch, self.n_slots, 3)


class TargetConditionedCircuitTokenDenoiser(nn.Module):
    def __init__(
        self,
        T: int = 200,
        n_slots: int = 6,
        target_dim: int = 32,
        time_dim: int = 64,
        hidden: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_mult: int = 4,
    ):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError("hidden must be divisible by num_heads")

        self.T = T
        self.n_slots = n_slots
        self.target_dim = target_dim
        self.time_dim = time_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.q_proj = nn.Linear(4, hidden)
        self.target_proj = nn.Linear(target_dim, hidden)
        self.time_proj = nn.Linear(time_dim, hidden)
        self.slot_embedding = nn.Embedding(n_slots, hidden)
        self.target_token = nn.Parameter(torch.zeros(hidden))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=ff_mult * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, q_stack: torch.Tensor, t_idx: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        if q_stack.ndim != 3 or q_stack.shape[1:] != (self.n_slots, 4):
            raise ValueError(f"Expected q_stack with shape (batch, {self.n_slots}, 4)")
        if target_features.ndim != 2 or target_features.shape != (q_stack.shape[0], self.target_dim):
            raise ValueError(f"Expected target_features with shape (batch, {self.target_dim})")

        batch = q_stack.shape[0]
        t_scaled = t_idx.float() / self.T
        temb = self.time_proj(timestep_embedding(t_scaled, self.time_dim))
        slot_ids = torch.arange(self.n_slots, device=q_stack.device)

        gate_tokens = self.q_proj(q_stack)
        gate_tokens = gate_tokens + self.slot_embedding(slot_ids)[None, :, :] + temb[:, None, :]
        target_token = self.target_proj(target_features) + temb + self.target_token[None, :]
        tokens = torch.cat([target_token[:, None, :], gate_tokens], dim=1)

        encoded = self.norm(self.encoder(tokens))
        return self.head(encoded[:, 1:, :]).reshape(batch, self.n_slots, 3)


class SkeletonConditionedCircuitTokenDenoiser(nn.Module):
    def __init__(
        self,
        T: int = 200,
        n_slots: int = 6,
        num_templates: int = 1,
        target_dim: int = 32,
        time_dim: int = 64,
        hidden: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_mult: int = 4,
        output_dim: int = 3,
    ):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
        if num_templates <= 0:
            raise ValueError("num_templates must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.T = T
        self.n_slots = n_slots
        self.num_templates = num_templates
        self.target_dim = target_dim
        self.time_dim = time_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.q_proj = nn.Linear(4, hidden)
        self.target_proj = nn.Linear(target_dim, hidden)
        self.time_proj = nn.Linear(time_dim, hidden)
        self.slot_embedding = nn.Embedding(n_slots, hidden)
        self.template_embedding = nn.Embedding(num_templates, hidden)
        self.active_embedding = nn.Embedding(2, hidden)
        self.target_token = nn.Parameter(torch.zeros(hidden))
        self.template_token = nn.Parameter(torch.zeros(hidden))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=ff_mult * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(
        self,
        q_stack: torch.Tensor,
        t_idx: torch.Tensor,
        target_features: torch.Tensor,
        template_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if q_stack.ndim != 3 or q_stack.shape[1:] != (self.n_slots, 4):
            raise ValueError(f"Expected q_stack with shape (batch, {self.n_slots}, 4)")
        if target_features.ndim != 2 or target_features.shape != (q_stack.shape[0], self.target_dim):
            raise ValueError(f"Expected target_features with shape (batch, {self.target_dim})")
        if template_ids.ndim != 1 or template_ids.shape[0] != q_stack.shape[0]:
            raise ValueError("Expected template_ids with shape (batch,)")
        if active_mask.shape != q_stack.shape[:2]:
            raise ValueError(f"Expected active_mask with shape (batch, {self.n_slots})")

        batch = q_stack.shape[0]
        active_mask = active_mask.to(device=q_stack.device, dtype=torch.bool)
        template_ids = template_ids.to(device=q_stack.device, dtype=torch.long)
        t_scaled = t_idx.float() / self.T
        temb = self.time_proj(timestep_embedding(t_scaled, self.time_dim))
        slot_ids = torch.arange(self.n_slots, device=q_stack.device)

        template_emb = self.template_embedding(template_ids)
        gate_tokens = self.q_proj(q_stack)
        gate_tokens = (
            gate_tokens
            + self.slot_embedding(slot_ids)[None, :, :]
            + self.active_embedding(active_mask.long())
            + template_emb[:, None, :]
            + temb[:, None, :]
        )
        target_token = self.target_proj(target_features) + template_emb + temb + self.target_token[None, :]
        template_token = template_emb + temb + self.template_token[None, :]
        tokens = torch.cat([target_token[:, None, :], template_token[:, None, :], gate_tokens], dim=1)

        encoded = self.norm(self.encoder(tokens))
        eps = self.head(encoded[:, 2:, :]).reshape(batch, self.n_slots, self.output_dim)
        return eps * active_mask[:, :, None].to(dtype=eps.dtype)


class HamiltonianSkeletonSelector(nn.Module):
    def __init__(
        self,
        target_dim: int = 129,
        num_templates: int = 4,
        hidden: int = 128,
    ):
        super().__init__()
        if num_templates <= 0:
            raise ValueError("num_templates must be positive")

        self.target_dim = target_dim
        self.num_templates = num_templates
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(target_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_templates),
        )

    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        if target_features.ndim != 2 or target_features.shape[1] != self.target_dim:
            raise ValueError(f"Expected target_features with shape (batch, {self.target_dim})")
        return self.net(target_features)


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
