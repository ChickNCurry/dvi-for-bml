import torch
from torch import Tensor, nn


class HyperNet(nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        non_linearity: str,
        num_steps: int,
        is_cross_attentive: bool = False,
        num_heads: int = 1,
    ) -> None:
        super(HyperNet, self).__init__()

        self.is_cross_attentive = is_cross_attentive
        self.num_heads = num_heads

        self.proj_t = nn.Embedding(num_steps + 1, h_dim)

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
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
        )

    def forward(self, t: int, context_embedding: Tensor, mask: Tensor | None) -> Tensor:
        # (1), (batch_size, h_dim) or (batch_size, context_size, h_dim), (batch_size, context_size)

        h: Tensor = self.proj_t(torch.tensor([t], device=context_embedding.device))
        # (1, h_dim)

        if self.is_cross_attentive:
            h = h.unsqueeze(1).expand(context_embedding.shape[0], -1, -1)
            # (batch_size, 1, h_dim)

            if mask is not None:
                mask = mask.unsqueeze(1)
                # (batch_size, 1, context_size)

                if self.num_heads > 1:
                    mask = mask.repeat(self.num_heads, 1, 1)
                    # (num_heads * batch_size, 1, context_size)

            h, _ = self.cross_attn(
                query=h,  # (batch_size, 1, h_dim)
                key=self.mlp_key(
                    context_embedding
                ),  # (batch_size, context_size, h_dim)
                value=self.mlp_value(
                    context_embedding
                ),  # (batch_size, context_size, h_dim)
                attn_mask=mask,  # (num_heads * batch_size, 1, context_size)
            )
            # (batch_size, 1, h_dim)

            h = h.squeeze(1)
            # (batch_size, h_dim)

        else:
            h = h + context_embedding
            # (batch_size, h_dim)

        hyper: Tensor = self.mlp(h)
        # (batch_size, z_dim)

        return hyper
