from typing import Tuple
import torch
from torch import Tensor, nn


class BCAControl(nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
    ) -> None:
        super(BCAControl, self).__init__()

        self.non_linearity = non_linearity

        self.proj_n = nn.Embedding(num_steps + 1, h_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)
        self.proj_z_mu = nn.Linear(h_dim, h_dim)
        self.proj_z_var = nn.Linear(h_dim, h_dim)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (
                    getattr(nn, non_linearity)(),
                    nn.Linear(h_dim, h_dim),
                )
                for _ in range(num_layers - 2)
            ],
            getattr(nn, non_linearity)()
        )

        self.proj_out = nn.Linear(h_dim, z_dim)

    def forward(
        self,
        n: int,
        z: Tensor,
        rest: Tuple[Tensor, Tensor],
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim),
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, context_size)

        z_mu, z_var = rest

        h: Tensor = (
            self.proj_n(torch.tensor([n], device=z.device))
            + self.proj_z(z)
            + self.proj_z_mu(z_mu)
            + self.proj_z_var(z_var)
        )
        # (batch_size, num_subtasks, h_dim)

        control_n: Tensor = self.proj_out(self.mlp(h))

        return control_n
