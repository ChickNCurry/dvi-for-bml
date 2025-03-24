from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor, nn
import torch


class AbstractEncoder(nn.Module, ABC):
    def __init__(
        self,
    ) -> None:
        super(AbstractEncoder, self).__init__()

    @abstractmethod
    def forward(
        self, context: Tensor, mask: Tensor | None
    ) -> Tuple[Tensor | Tuple[Tensor, Tensor], Tensor]:
        pass

    def compute_h(self, context: Tensor, mask: Tensor | None) -> Tensor:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        batch_size = context.shape[0]
        num_subtasks = context.shape[1]
        data_size = context.shape[2]

        h: Tensor = self.proj_in(context)
        # (batch_size, num_subtasks, data_size, h_dim)

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

    def compute_s(self, context: Tensor, mask: Tensor | None) -> Tensor:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        if mask is None:
            return (
                torch.tensor([context.shape[2]], device=context.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(context.shape[0], context.shape[1])
            )  # (batch_size, num_subtasks)
        else:
            return mask.sum(dim=-1)
            # (batch_size, num_subtasks)
