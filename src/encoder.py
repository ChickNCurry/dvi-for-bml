from typing import Tuple

import torch
from torch import Tensor, nn


class SetEncoder(nn.Module):
    def __init__(
        self,
        c_dim: int,
        h_dim: int,
        num_layers: int,
        non_linearity: str,
        is_attentive: bool,
        is_aggregative: bool,
        is_non_aggregative: bool,
        use_context_size: bool,
        aggregation: str,
        max_context_size: int,
    ) -> None:
        super(SetEncoder, self).__init__()

        self.is_attentive = is_attentive
        self.is_aggregative = is_aggregative
        self.is_non_aggregative = is_non_aggregative
        self.use_context_size = use_context_size

        self.proj_in = nn.Linear(c_dim, h_dim)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            *[
                layer
                for layer in (
                    getattr(nn, non_linearity)(),
                    nn.Linear(h_dim, h_dim),
                )
                for _ in range(num_layers - 1)
            ],
        )

        if self.is_attentive:
            self.self_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

        if self.is_aggregative:
            self.aggregation = aggregation

            if self.use_context_size:
                self.context_size_embedding = nn.Embedding(max_context_size + 1, h_dim)

    def forward(
        self, context: Tensor, mask: Tensor | None
    ) -> Tuple[Tensor | None, Tensor | None]:
        # (batch_size, context_size, c_dim), (batch_size, context_size)

        c: Tensor = self.proj_in(context)
        # (batch_size, context_size, h_dim)

        if self.is_attentive:
            c, _ = self.self_attn(c, c, c, need_weights=False)
            # (batch_size, context_size, h_dim)

        c = self.mlp(c)
        # (batch_size, context_size, h_dim)

        aggregated: Tensor | None = None
        non_aggregated: Tensor | None = None

        if self.is_aggregative:
            match self.aggregation:
                case "mean":
                    aggregated = (
                        c.mean(dim=1)
                        if mask is None
                        else (c * mask.unsqueeze(-1)).sum(dim=1)
                        / mask.sum(dim=1, keepdim=True)
                    )
                case "max":
                    aggregated = (
                        c.max(dim=1)[0]
                        if mask is None
                        else (c * mask.unsqueeze(-1)).max(dim=1)[0]
                    )
                # (batch_size, h_dim)

            if self.use_context_size:
                e = (
                    self.context_size_embedding(
                        torch.tensor([context.shape[1]], device=c.device)
                    ).expand(c.shape[0], -1)
                    if mask is None
                    else self.context_size_embedding(mask.sum(dim=1).int())
                )
                # (batch_size, h_dim)

                aggregated = aggregated + e if aggregated is not None else None
                # (batch_size, h_dim)

        if self.is_non_aggregative:
            non_aggregated = c if mask is None else c * mask.unsqueeze(-1)
            # (batch_size, context_size, h_dim)

        return aggregated, non_aggregated


class TestEncoder(nn.Module):
    def __init__(self, c_dim: int, h_dim: int) -> None:
        super(TestEncoder, self).__init__()

        self.proj_c = nn.Linear(c_dim, h_dim)

    def forward(self, c: Tensor, mask: Tensor | None) -> Tuple[Tensor, None]:

        c = self.proj_c(c.squeeze(1))

        return c, None
