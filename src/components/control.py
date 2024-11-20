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
    ) -> None:
        super(Control, self).__init__()

        self.is_cross_attentive = is_cross_attentive

        self.proj_t = nn.Embedding(num_steps + 1, h_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)

        self.mlp_before = nn.Sequential(
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

        self.mlp_after = nn.Sequential(
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

        if self.is_cross_attentive:
            self.cross_attn = nn.MultiheadAttention(
                h_dim, num_heads=1, batch_first=True
            )

        self.proj_out = nn.Linear(h_dim, z_dim)

    def forward(
        self, z: Tensor, t: int, context_embedding: Tensor, mask: Tensor | None
    ) -> Tensor:
        # (batch_size, z_dim),
        # (1),
        # (batch_size, h_dim) or (batch_size, context_size, h_dim)
        # (batch_size, context_size)

        z = self.proj_z(z)
        t = self.proj_t(torch.tensor([t], device=z.device))
        # (batch_size, h_dim)

        if self.is_cross_attentive:
            h: Tensor = self.mlp_before(z + t)
            # (batch_size, h_dim)

            h, _ = self.cross_attn(
                query=h.unsqueeze(1),  # (batch_size, 1, h_dim)
                key=context_embedding,  # (batch_size, context_size, h_dim)
                value=context_embedding,  # (batch_size, context_size, h_dim)
                attn_mask=(
                    mask.unsqueeze(1) if mask is not None else None
                ),  # (batch_size, 1, context_size)
            )
            # (batch_size, h_dim)

            control: Tensor = self.proj_out(self.mlp_after(h.squeeze(1)))
            # (batch_size, z_dim)
        else:
            control = self.proj_out(self.mlp_before(z + t + context_embedding))
            # (batch_size, z_dim)

        return control
