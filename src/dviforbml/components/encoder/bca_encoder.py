from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from dviforbml.components.encoder.abstract_encoder import AbstractEncoder


class BCAEncoder(AbstractEncoder):
    def __init__(
        self,
        c_dim: int,
        h_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
        num_heads: int | None,
        max_context_size: int | None,
    ) -> None:
        super(BCAEncoder, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_heads = num_heads
        self.max_context_size = max_context_size

        self.proj_in = nn.Linear(c_dim, h_dim)

        if num_heads is not None:
            self.self_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers)
            ],
        )

        self.proj_r = nn.Linear(h_dim, z_dim)
        self.proj_r_var = nn.Linear(h_dim, z_dim)

    def forward(
        self, context: Tensor, mask: Tensor | None
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        h = self.compute_h(context, mask)
        # (batch_size, num_subtasks, context_size, h_dim)

        batch_size = context.shape[0]
        num_subtasks = context.shape[1]

        r = self.proj_r(h)
        r_var = softplus(torch.clamp(self.proj_r_var(h), min=1e-6, max=1e2))
        # (batch_size, num_subtasks, context_size, h_dim)

        z_var_0 = torch.ones((batch_size, num_subtasks, self.z_dim), device=h.device)
        z_mu_0 = torch.zeros((batch_size, num_subtasks, self.z_dim), device=h.device)
        # (batch_size, num_subtasks, h_dim)

        if mask is None:
            z_var = -(-z_var_0 + torch.sum(-r_var, dim=2))
            z_mu = z_mu_0 + z_var * torch.sum(
                (r - z_mu_0[:, :, None, :]) / r_var, dim=2
            )  # (batch_size, num_subtasks, h_dim)
        else:
            z_var = -(-z_var_0 + torch.sum(-r_var * mask.unsqueeze(-1), dim=2))
            z_mu = z_mu_0 + z_var * torch.sum(
                ((r - z_mu_0[:, :, None, :]) / r_var) * mask.unsqueeze(-1), dim=2
            )  # (batch_size, num_subtasks, h_dim)

        s = self.compute_s(context, mask) if self.max_context_size is not None else None
        # (batch_size, num_subtasks)

        return (z_mu, z_var), s
