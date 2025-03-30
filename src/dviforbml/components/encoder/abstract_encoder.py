from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor, nn
import torch


class AbstractEncoder(nn.Module, ABC):
    def __init__(
        self,
        c_dim: int,
        h_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
        num_heads: int | None,
        num_blocks: int,
        max_context_size: int | None,
    ) -> None:
        super(AbstractEncoder, self).__init__()

        self.h_dim = h_dim
        self.num_heads = num_heads
        self.max_context_size = max_context_size

        self.proj_in = nn.Linear(c_dim, h_dim)

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(h_dim, num_layers, non_linearity, num_heads)
                for _ in range(num_blocks)
            ]
        )

        if max_context_size is not None:
            self.proj_s = nn.Embedding(max_context_size + 1, z_dim)

    def compute_h(self, context: Tensor, mask: Tensor | None) -> Tensor:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        h: Tensor = self.proj_in(context)
        # (batch_size, num_subtasks, data_size, h_dim)

        for block in self.blocks:
            h = block(h, mask)
            # (batch_size, num_subtasks, data_size, h_dim)

        return h

    def compute_s(self, context: Tensor, mask: Tensor | None) -> Tensor:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        batch_size = context.shape[0]
        num_subtasks = context.shape[1]
        data_size = context.shape[2]

        if mask is None:
            s = torch.tensor([data_size], device=context.device, dtype=torch.int)
            s = s.repeat(batch_size, num_subtasks)
            # (batch_size, num_subtasks)
        else:
            s = mask.sum(dim=-1).int()
            # (batch_size, num_subtasks)

        s = self.proj_s(s)
        # (batch_size, num_subtasks, z_dim)

        return s

    @abstractmethod
    def forward(
        self, context: Tensor, mask: Tensor | None
    ) -> Tuple[Tensor | Tuple[Tensor, Tensor], Tensor]:
        pass


class EncoderBlock(nn.Module):
    def __init__(
        self,
        h_dim: int,
        num_layers: int,
        non_linearity: str,
        num_heads: int | None,
    ):
        super(EncoderBlock, self).__init__()

        self.num_heads = num_heads

        if num_heads is not None:
            self.self_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers)
            ],
        )

    def forward(self, h: Tensor, mask: Tensor | None) -> Tensor:
        # (batch_size, num_subtasks, data_size, h_dim)
        # (batch_size, num_subtasks, data_size)

        batch_size = h.shape[0]
        num_subtasks = h.shape[1]
        data_size = h.shape[2]

        if self.num_heads is not None:
            h = h.view(batch_size * num_subtasks, data_size, -1)
            # (batch_size * num_subtasks, data_size, h_dim)

            key_padding_mask = (
                (mask.view(batch_size * num_subtasks, -1).bool().logical_not())
                if mask is not None
                else None
            )  # (batch_size * num_subtasks, data_size)

            h, _ = self.self_attn(
                query=h,
                key=h,
                value=h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )  # (batch_size * num_subtasks, data_size, h_dim)

            h = h.view(batch_size, num_subtasks, data_size, -1)
            # (batch_size, num_subtasks, data_size, h_dim)

        h = self.mlp(h)
        # (batch_size, num_subtasks, data_size, h_dim)

        return h
