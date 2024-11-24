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
        num_heads: int = 1,
    ) -> None:
        super(Control, self).__init__()

        self.is_cross_attentive = is_cross_attentive

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
                h_dim, num_heads=num_heads, batch_first=True
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
            h: Tensor = (z + t).unsqueeze(1)
            # (batch_size, 1, h_dim)

            h, _ = self.cross_attn(
                query=h,  # (batch_size, 1, h_dim)
                key=self.mlp_key(
                    context_embedding
                ),  # (batch_size, context_size, h_dim)
                value=self.mlp_value(
                    context_embedding
                ),  # (batch_size, context_size, h_dim)
                attn_mask=(
                    mask.unsqueeze(1) if mask is not None else None
                ),  # (batch_size, 1, context_size)
            )
            # (batch_size, 1, h_dim)

            h = h.squeeze(1)
            # (batch_size, h_dim)
        else:
            h = z + t + context_embedding
            # (batch_size, h_dim)

        control: Tensor = self.proj_out(self.mlp(h))
        # (batch_size, z_dim)

        return control
