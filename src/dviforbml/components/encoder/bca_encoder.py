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
        num_blocks: int,
        max_context_size: int | None,
        bca_dim: int,
    ) -> None:
        super(BCAEncoder, self).__init__(
            c_dim,
            h_dim,
            z_dim,
            num_layers,
            non_linearity,
            num_heads,
            num_blocks,
            max_context_size,
        )

        self.bca_dim = bca_dim

        self.proj_r = nn.Linear(h_dim, bca_dim)
        self.proj_r_var = nn.Linear(h_dim, bca_dim)

    def forward(
        self, context: Tensor, mask: Tensor | None
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        batch_size = context.shape[0]
        num_subtasks = context.shape[1]

        h = self.compute_h(context, mask)
        # (batch_size, num_subtasks, context_size, h_dim)

        r = self.proj_r(h)
        r_var = softplus(torch.clamp(self.proj_r_var(h), min=1e-6, max=1e2))
        # (batch_size, num_subtasks, context_size, bca_dim)

        z_var_0 = torch.ones((batch_size, num_subtasks, self.bca_dim), device=h.device)
        z_mu_0 = torch.zeros((batch_size, num_subtasks, self.bca_dim), device=h.device)
        # (batch_size, num_subtasks, bca_dim)

        if mask is None:
            z_var = -(-z_var_0 + torch.sum(-r_var, dim=2))
            z_mu = z_mu_0 + z_var * torch.sum(
                (r - z_mu_0[:, :, None, :]) / r_var, dim=2
            )  # (batch_size, num_subtasks, bca_dim)
        else:
            z_var = -(-z_var_0 + torch.sum(-r_var * mask.unsqueeze(-1), dim=2))
            z_mu = z_mu_0 + z_var * torch.sum(
                ((r - z_mu_0[:, :, None, :]) / r_var) * mask.unsqueeze(-1), dim=2
            )  # (batch_size, num_subtasks, bca_dim)

        s = self.compute_s(context, mask) if self.max_context_size is not None else None
        # (batch_size, num_subtasks, z_dim)

        return (z_mu, z_var), s
