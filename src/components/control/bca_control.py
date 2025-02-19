from typing import Tuple

import torch
from torch import Tensor, nn

from src.components.control.base_control import BaseControl


class BCAControl(BaseControl):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
        use_score: bool,
    ) -> None:
        super(BCAControl, self).__init__()

        self.non_linearity = non_linearity
        self.use_score = use_score

        self.proj_n = nn.Embedding(num_steps + 1, h_dim)
        # self.proj_z = nn.Linear(z_dim, h_dim)
        # self.proj_z_mu = nn.Linear(h_dim, h_dim)
        # self.proj_z_var = nn.Linear(h_dim, h_dim)

        self.mlp = nn.Sequential(
            nn.Linear(z_dim + 3 * h_dim, h_dim),
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

        if self.use_score:
            self.proj_offset = nn.Linear(h_dim, z_dim)

            nn.init.zeros_(self.proj_offset.weight)
            nn.init.zeros_(self.proj_offset.bias)

            self.proj_scale = nn.Linear(h_dim, z_dim)

            nn.init.zeros_(self.proj_scale.weight)
            nn.init.ones_(self.proj_scale.bias)
        else:
            self.proj_out = nn.Linear(h_dim, z_dim)

    def forward(
        self,
        n: int,
        z: Tensor,
        r: Tensor | Tuple[Tensor, Tensor],
        mask: Tensor | None,
        score: Tensor | None,
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim),
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, context_size)

        if self.use_score:
            assert score is not None

        z_mu, z_var = r
        n_emb = self.proj_n(torch.tensor([n], device=z.device))
        input = torch.cat([z, z_mu, z_var, n_emb], dim=-1)
        # (batch_size, num_subtasks, h_dim)

        h = self.mlp(input)
        # (batch_size, num_subtasks, h_dim)

        if self.use_score:
            offset: Tensor = self.proj_offset(h)
            scale: Tensor = self.proj_scale(h)
            # (batch_size, num_subtasks, z_dim)

            control_n = offset + scale * score
            # (batch_size, num_subtasks, z_dim)
        else:
            control_n = self.proj_out(h)
            # (batch_size, num_subtasks, z_dim)

        return control_n
