from typing import Tuple
from torch import Tensor, nn

from dviforbml.components.encoder.abstract_encoder import AbstractEncoder


class MHCAEncoder(AbstractEncoder):
    def __init__(
        self,
        c_dim: int,
        h_dim: int,
        num_layers: int,
        non_linearity: str,
        num_heads: int | None,
        max_context_size: int | None,
    ) -> None:
        super(MHCAEncoder, self).__init__()

        self.h_dim = h_dim
        self.num_heads = num_heads
        self.max_context_size = max_context_size

        self.proj_in = nn.Linear(c_dim, h_dim)

        if self.num_heads is not None:
            self.self_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers)
            ],
        )

    def forward(self, context: Tensor, mask: Tensor | None) -> Tuple[Tensor, Tensor]:
        h = self.compute_h(context, mask)
        # (batch_size, num_subtasks, context_size, h_dim)

        r = h if mask is None else h * mask.unsqueeze(-1)
        # (batch_size, num_subtasks, context_size, h_dim)

        s = self.compute_s(context, mask) if self.max_context_size is not None else None
        # (batch_size, num_subtasks)

        return r, s
