from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import torch
from torch import Tensor, nn


class Aggr(Enum):
    MEAN = "mean"
    MAX = "max"


class Encoder(nn.Module, ABC):
    def __init__(
        self,
    ) -> None:
        super(Encoder, self).__init__()

    @abstractmethod
    def forward(
        self, context: Tensor, mask: Tensor | None
    ) -> Tuple[Tensor | None, Tensor | None]:
        pass


class SetEncoder(Encoder):
    def __init__(
        self,
        c_dim: int,
        h_dim: int,
        num_layers: int,
        non_linearity: str,
        is_attentive: bool,
        num_heads: int | None,
        is_non_aggregative: bool,
        is_aggregative: bool,
        aggregation: Aggr | None,
        use_context_size_emb: bool,
        max_context_size: int | None,
    ) -> None:
        super(SetEncoder, self).__init__()

        self.is_attentive = is_attentive
        self.is_aggregative = is_aggregative
        self.is_non_aggregative = is_non_aggregative
        self.use_context_size_emb = use_context_size_emb

        self.proj_in = nn.Linear(c_dim, h_dim)

        if self.is_attentive:
            assert num_heads is not None
            self.self_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 1)
            ],
        )

        if self.is_aggregative:
            self.aggregation = aggregation

            if self.use_context_size_emb:
                assert max_context_size is not None
                self.context_size_embed = nn.Embedding(max_context_size + 1, h_dim)

    def forward(
        self, context: Tensor, mask: Tensor | None
    ) -> Tuple[Tensor | None, Tensor | None]:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)

        batch_size = context.shape[0]
        num_subtasks = context.shape[1]
        context_size = context.shape[2]

        h: Tensor = self.proj_in(context)
        # (batch_size, num_subtasks, context_size, h_dim)

        if self.is_attentive:
            h = h.view(batch_size * num_subtasks, context_size, -1)
            # (batch_size * num_subtasks, context_size, h_dim)

            h, _ = self.self_attn(h, h, h, need_weights=False)

            h = h.view(batch_size, num_subtasks, context_size, -1)
            # (batch_size, num_subtasks, context_size, h_dim)

        h = self.mlp(h)
        # (batch_size, num_subtasks, context_size, h_dim)

        r_aggr: Tensor | None = None
        r_non_aggr: Tensor | None = None

        if self.is_aggregative:

            match self.aggregation:
                case Aggr.MEAN:

                    if mask is None:
                        r_aggr = h.mean(dim=2)
                    else:
                        r_aggr = (h * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(
                            dim=2, keepdim=True
                        )

                case Aggr.MAX:

                    if mask is None:
                        r_aggr = h.max(dim=2)[0]
                    else:
                        r_aggr = (h * mask.unsqueeze(-1)).max(dim=2)[0]

                # (batch_size, num_subtasks, h_dim)

            if self.use_context_size_emb:

                if mask is None:
                    e = self.context_size_embed(
                        torch.tensor([context.shape[2]], device=h.device)
                    ).expand(h.shape[0], h.shape[1], -1)
                else:
                    e = self.context_size_embed(mask.sum(dim=2).int())

                # (batch_size, num_subtasks, h_dim)

                r_aggr = r_aggr + e if r_aggr is not None else None
                # (batch_size, num_subtasks, h_dim)

        if self.is_non_aggregative:
            r_non_aggr = h if mask is None else h * mask.unsqueeze(-1)
            # (batch_size, num_subtasks, context_size, h_dim)

        return r_aggr, r_non_aggr


class SingletonEncoder(Encoder):
    def __init__(self, c_dim: int, h_dim: int) -> None:
        super(SingletonEncoder, self).__init__()

        self.proj_c = nn.Linear(c_dim, h_dim)

    def forward(self, context: Tensor, mask: Tensor | None) -> Tuple[Tensor, None]:
        # (batch_size, num_subtasks, context_size, c_dim)

        assert context.shape[2] == 1

        r_aggr: Tensor = self.proj_c(context.squeeze(2))
        # (batch_size, num_subtasks, h_dim)

        return r_aggr, None
