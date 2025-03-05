import torch
from torch import Tensor, nn

from src.components.control.abstract_control import AbstractControl


class AggrControl(AbstractControl):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
        use_score: bool,
        use_error: bool,
    ) -> None:
        super(AggrControl, self).__init__()

        self.non_linearity = non_linearity
        self.use_score = use_score
        self.use_error = use_error

        self.proj_n = nn.Embedding(num_steps + 1, h_dim)

        self.mlp = nn.Sequential(
            nn.Linear(z_dim + 2 * h_dim + (1 if self.use_error else 0), h_dim),
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers)
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
        r: Tensor,
        mask: Tensor | None,
        score: Tensor | None,
        error: Tensor | None,
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks)

        if self.use_score:
            assert score is not None

        n_emb = self.proj_n(torch.tensor([n], device=z.device)).repeat(
            z.shape[0], z.shape[1], 1
        )  # (batch_size, num_subtasks, h_dim)

        input = torch.cat([z, r, n_emb], dim=-1)
        # (batch_size, num_subtasks, z_dim + 2 * h_dim)

        if self.use_error:
            assert error is not None
            input = torch.cat([input, error], dim=-1)
            # (batch_size, num_subtasks, z_dim + 2 * h_dim + 1)

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
