from typing import Tuple

import torch
from torch import Tensor, nn

from dviforbml.components.control.abstract_control import AbstractControl


class MHCAControl(AbstractControl):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
        max_context_size: int | None,
        use_score: bool,
        num_heads: int,
    ) -> None:
        super(MHCAControl, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.max_context_size = max_context_size
        self.use_score = use_score

        self.proj_n = nn.Embedding(num_steps + 1, z_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)

        self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        input_dim = z_dim + z_dim + (z_dim if max_context_size is not None else 0)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            *[
                layer
                for _ in range(num_layers)
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
            ],
            getattr(nn, non_linearity)(),
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
        s: Tensor | None,
        score: Tensor | None,
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim),
        # (batch_size, num_subtasks, data_size, h_dim)
        # (batch_size, num_subtasks, data_size)
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, z_dim)

        if self.use_score:
            assert score is not None

        batch_size = r.shape[0]
        num_subtasks = r.shape[1]
        data_size = r.shape[2]

        n_emb = self.proj_n(torch.tensor([n], device=z.device)).repeat(
            batch_size, num_subtasks, 1
        )  # (batch_size, num_subtasks, z_dim)

        input = torch.cat([z, n_emb], dim=-1)
        # (batch_size, num_subtasks, h_dim + z_dim)

        if self.max_context_size is not None:
            assert s is not None
            input = torch.cat([input, s], dim=-1)
            # (batch_size, num_subtasks, h_dim + 2 * z_dim)

        h = self.mlp(input)
        # (batch_size, num_subtasks, h_dim)

        r = r.view(batch_size * num_subtasks, data_size, self.h_dim)
        # (batch_size * num_subtasks, data_size, h_dim)

        # z = self.proj_z(z).view(batch_size * num_subtasks, -1).unsqueeze(1)
        # # (batch_size * num_subtasks, 1, h_dim)

        h = h.view(batch_size * num_subtasks, self.h_dim).unsqueeze(1)
        # (batch_size * num_subtasks, 1, h_dim)

        key_padding_mask = (
            mask.view(batch_size * num_subtasks, data_size).bool().logical_not()
            if mask is not None
            else None
        )  # (batch_size * num_subtasks, data_size)

        h, _ = self.cross_attn(
            query=h, key=r, value=r, key_padding_mask=key_padding_mask
        )  # (batch_size * num_subtasks, 1, h_dim)

        h = h.reshape(batch_size, num_subtasks, self.h_dim)
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
