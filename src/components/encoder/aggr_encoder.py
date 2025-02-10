from enum import Enum

import torch
from torch import Tensor, nn

from src.components.encoder.base_encoder import BaseEncoder


class Aggr(Enum):
    MEAN = "mean"
    MAX = "max"


class AggrEncoder(BaseEncoder):
    def __init__(
        self,
        c_dim: int,
        h_dim: int,
        num_layers: int,
        non_linearity: str,
        num_heads: int | None,
        aggregation: Aggr | None,
        max_context_size: int | None,
    ) -> None:
        super(AggrEncoder, self).__init__()

        self.h_dim = h_dim
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.max_context_size = max_context_size

        self.proj_in = nn.Linear(c_dim, h_dim)

        if num_heads is not None:
            self.self_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 1)
            ],
        )

        if max_context_size is not None:
            self.context_size_embed = nn.Embedding(max_context_size + 1, h_dim)

    def forward(self, context: Tensor, mask: Tensor | None) -> Tensor:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)

        batch_size = context.shape[0]
        num_subtasks = context.shape[1]
        context_size = context.shape[2]

        h: Tensor = self.proj_in(context)
        # (batch_size, num_subtasks, context_size, h_dim)

        if self.num_heads is not None:
            h = h.view(batch_size * num_subtasks, context_size, -1)
            # (batch_size * num_subtasks, context_size, h_dim)

            key_padding_mask = (
                (mask.view(batch_size * num_subtasks, -1).bool().logical_not())
                if mask is not None
                else None
            )  # (batch_size * num_subtasks, context_size)

            h, _ = self.self_attn(
                query=h,
                key=h,
                value=h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )  # (batch_size * num_subtasks, context_size, h_dim)

            h = h.view(batch_size, num_subtasks, context_size, -1)
            # (batch_size, num_subtasks, context_size, h_dim)

        h = self.mlp(h)
        # (batch_size, num_subtasks, context_size, h_dim)

        match self.aggregation:
            case Aggr.MEAN:
                if mask is None:
                    r = h.mean(dim=2)
                else:
                    r = (h * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(
                        dim=2, keepdim=True
                    )
            case Aggr.MAX:
                if mask is None:
                    r = h.max(dim=2)[0]
                else:
                    r = (h * mask.unsqueeze(-1)).max(dim=2)[0]
            # (batch_size, num_subtasks, h_dim)

        if self.max_context_size is not None:
            if mask is None:
                e = self.context_size_embed(
                    torch.tensor([context.shape[2]], device=h.device)
                ).expand(h.shape[0], h.shape[1], -1)
            else:
                e = self.context_size_embed(mask.sum(dim=2).int())
            r = r + e
            # (batch_size, num_subtasks, h_dim)

        return r
