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
        is_cross_attentive: bool,
        num_heads: int | None,
    ) -> None:
        super(Control, self).__init__()

        self.is_cross_attentive = is_cross_attentive
        self.num_heads = num_heads

        self.proj_t = nn.Embedding(num_steps + 1, h_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)

        if self.is_cross_attentive:
            assert self.num_heads is not None
            self.cross_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 2)
            ],
            getattr(nn, non_linearity)()
        )

        self.proj_control = nn.Linear(h_dim, z_dim)

    def forward(
        self,
        t: int,
        z: Tensor,
        r_aggr: Tensor | None,
        r_non_aggr: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim),
        # (batch_size, num_subtasks, h_dim)
        # (batch_size, num_subtasks, context_size, h_dim)
        # (batch_size, num_subtasks, context_size)

        batch_size = z.shape[0]
        num_subtasks = z.shape[1]

        h: Tensor = self.proj_t(torch.tensor([t], device=z.device)) + self.proj_z(z)
        # (batch_size, num_subtasks, h_dim)

        if self.is_cross_attentive:
            assert r_non_aggr is not None and self.num_heads is not None

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
                query=h, key=r_non_aggr, value=r_non_aggr, attn_mask=mask
            )  # (batch_size, num_subtasks, 1, h_dim)

            h = h.squeeze(2)
            # (batch_size, num_subtasks, h_dim)

        else:
            assert r_aggr is not None

            h = h + r_aggr
            # (batch_size, num_subtasks, h_dim)

        control_t: Tensor = self.proj_control(self.mlp(h))
        # (batch_size, num_subtasks, z_dim)

        return control_t
