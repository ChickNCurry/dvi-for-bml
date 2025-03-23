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
    ) -> None:
        super(BCAEncoder, self).__init__()

        self.num_heads = num_heads
        self.h_dim = h_dim
        self.z_dim = z_dim

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

    def forward(self, context: Tensor, mask: Tensor | None) -> Tuple[Tensor, Tensor]:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)

        batch_size = context.shape[0]
        num_subtasks = context.shape[1]
        context_size = context.shape[2]

        h: Tensor = self.proj_in(context)
        # (batch_size, num_subtasks, context_size, h_dim)

        if self.num_heads is not None:
            h = h.view(batch_size * num_subtasks, context_size, -1)
            # (batch_size * num_subtasks, context_size, h_dim)

            key_padding_mask = (
                (mask.view(batch_size * num_subtasks, -1).bool().logical_not())
                if mask is not None
                else None
            )  # (batch_size * num_subtasks, context_size)

            h, _ = self.self_attn(
                query=h,
                key=h,
                value=h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )  # (batch_size * num_subtasks, context_size, h_dim)

            h = h.view(batch_size, num_subtasks, context_size, -1)
            # (batch_size, num_subtasks, context_size, h_dim)

        h = self.mlp(h)

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

        return z_mu, z_var
