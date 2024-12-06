import torch
from torch import Tensor, nn


class Control(nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
        num_steps: int,
        is_cross_attentive: bool,
        num_heads: int,
    ) -> None:
        super(Control, self).__init__()

        self.is_cross_attentive = is_cross_attentive
        self.num_heads = num_heads

        self.proj_t = nn.Embedding(num_steps + 1, h_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)

        if self.is_cross_attentive:
            self.mlp_key = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                getattr(nn, non_linearity)(),
                nn.Linear(h_dim, h_dim),
            )
            self.mlp_value = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                getattr(nn, non_linearity)(),
                nn.Linear(h_dim, h_dim),
            )
            self.cross_attn = nn.MultiheadAttention(
                h_dim, num_heads=self.num_heads, batch_first=True
            )

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            *[
                layer
                for layer in (
                    getattr(nn, non_linearity)(),
                    nn.Linear(h_dim, h_dim),
                )
                for _ in range(num_layers - 1)
            ]
        )

        self.proj_out = nn.Linear(h_dim, z_dim)

    def forward(
        self, t: int, z: Tensor, context_embedding: Tensor, mask: Tensor | None
    ) -> Tensor:
        # (batch_size, num_subtasks, z_dim),
        # (batch_size, num_subtasks, h_dim) or (batch_size, num_subtasks, context_size, h_dim)
        # (batch_size, num_subtasks, context_size)

        t = self.proj_t(torch.tensor([t], device=z.device))
        z = self.proj_z(z)
        # (batch_size, num_subtasks, h_dim)

        h: Tensor = t + z
        # (batch_size, num_subtasks, h_dim)

        if self.is_cross_attentive:
            h = h.unsqueeze(2)
            # (batch_size, num_subtasks, 1, h_dim)

            if mask is not None:
                mask = mask.unsqueeze(2)
                # (batch_size, num_subtasks, 1, context_size)

                if self.num_heads > 1:
                    mask = mask.repeat(self.num_heads, 1, 1, 1)
                    # (num_heads * batch_size, num_subtasks, 1, context_size)

            key = self.mlp_key(context_embedding)
            value = self.mlp_value(context_embedding)
            # (batch_size, num_subtasks, context_size, h_dim)

            h, _ = self.cross_attn(query=h, key=key, value=value, attn_mask=mask)
            # (batch_size, num_subtasks, 1, h_dim)

            h = h.squeeze(2)
            # (batch_size, num_subtasks, h_dim)
        else:
            h = h + context_embedding
            # (batch_size, num_subtasks, h_dim)

        control: Tensor = self.proj_out(self.mlp(h))
        # (batch_size, num_subtasks, z_dim)

        return control
