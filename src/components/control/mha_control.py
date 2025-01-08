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
        num_heads: int,
    ) -> None:
        super(Control, self).__init__()

        self.num_heads = num_heads
        self.non_linearity = non_linearity

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

        self.proj_out = nn.Linear(h_dim, z_dim)

    def forward(
        self,
        n: int,
        z: Tensor,
        rest: Tuple[Tensor, Tensor | None],
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim),
        # (batch_size, num_subtasks, context_size, h_dim)
        # (batch_size, num_subtasks, context_size)

        r, mask = rest

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

        control_n: Tensor = self.proj_out(self.mlp(h))
        # (batch_size, num_subtasks, h_dim)

        return control_n
