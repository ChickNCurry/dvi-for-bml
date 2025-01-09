from typing import Tuple
import torch
from torch import Tensor, nn


class Control(nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        num_steps: int,
        num_layers: int,
        non_linearity: str,
        uses_score: bool,
        num_heads: int,
    ) -> None:
        super(Control, self).__init__()

        self.non_linearity = non_linearity
        self.uses_score = uses_score
        self.num_heads = num_heads

        self.proj_n = nn.Embedding(num_steps + 1, h_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)

        self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

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

        if self.uses_score:
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
        mask: Tensor,
        score: Tensor | None,
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim),
        # (batch_size, num_subtasks, context_size, h_dim)
        # (batch_size, num_subtasks, context_size)

        if self.uses_score:
            assert score is not None

        batch_size = z.shape[0]
        num_subtasks = z.shape[1]

        h: Tensor = self.proj_n(torch.tensor([n], device=z.device)) + self.proj_z(z)
        # (batch_size, num_subtasks, h_dim)

        h = h.unsqueeze(2)
        # (batch_size, num_subtasks, 1, h_dim)

        mask = (
            mask.unsqueeze(2)
            .view(batch_size * num_subtasks, -1, -1)
            .repeat(self.num_heads, 1, 1)
            if mask is not None
            else None
        )  # (num_heads * num_subtasks * batch_size, 1, context_size)

        h, _ = self.cross_attn(
            query=h, key=r, value=r, attn_mask=mask
        )  # (batch_size, num_subtasks, 1, h_dim)

        h = h.squeeze(2)
        # (batch_size, num_subtasks, h_dim)

        h = self.mlp(h)
        # (batch_size, num_subtasks, h_dim)

        if self.uses_score:
            offset: Tensor = self.proj_offset(h)
            scale: Tensor = self.proj_scale(h)
            # (batch_size, num_subtasks, z_dim)

            control_n = offset + scale * score
            # (batch_size, num_subtasks, z_dim)
        else:
            control_n = self.proj_out(h)
            # (batch_size, num_subtasks, z_dim)

        return control_n
