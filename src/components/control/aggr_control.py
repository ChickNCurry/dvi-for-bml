from typing import Tuple
import torch
from torch import Tensor, nn

from src.components.control.base_control import BaseControl


class AggrControl(BaseControl):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
        use_score: bool,
    ) -> None:
        super(AggrControl, self).__init__()

        self.non_linearity = non_linearity
        self.use_score = use_score

        self.proj_n = nn.Embedding(num_steps + 1, h_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (
                    # nn.BatchNorm1d(h_dim),
                    getattr(nn, non_linearity)(),
                    nn.Linear(h_dim, h_dim),
                )
                for _ in range(num_layers - 2)
            ],
            # nn.BatchNorm1d(h_dim),
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
        r: Tensor,
        mask: None,
        score: Tensor | None,
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim),
        # (batch_size, num_subtasks, h_dim)

        if self.use_score:
            assert score is not None

        h: Tensor = self.proj_n(torch.tensor([n], device=z.device)) + self.proj_z(z)
        # (batch_size, num_subtasks, h_dim)

        h = h + r
        # (batch_size, num_subtasks, h_dim)

        # h = h.view(batch_size * num_subtasks, -1)
        # z = z.view(batch_size * num_subtasks, -1)
        # (batch_size * num_subtasks, h_dim)

        # control_t: Tensor = getattr(nn, self.non_linearity)()(
        #     self.proj_out(self.mlp(h)) + z
        # )  # (batch_size * num_subtasks, z_dim)

        # control_t = control_t.view(batch_size, num_subtasks, -1)
        # (batch_size, num_subtasks, z_dim)

        h = self.mlp(h)
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
