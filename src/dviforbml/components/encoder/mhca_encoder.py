from typing import Tuple
from torch import Tensor

from dviforbml.components.encoder.abstract_encoder import AbstractEncoder


class MHCAEncoder(AbstractEncoder):
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
        super(MHCAEncoder, self).__init__(
            c_dim,
            h_dim,
            z_dim,
            num_layers,
            non_linearity,
            num_heads,
            num_blocks,
            max_context_size,
        )

    def forward(self, context: Tensor, mask: Tensor | None) -> Tuple[Tensor, Tensor]:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        h = self.compute_h(context, mask)
        # (batch_size, num_subtasks, context_size, h_dim)

        s = self.compute_s(context, mask) if self.max_context_size is not None else None
        # (batch_size, num_subtasks, z_dim)

        return h, s
