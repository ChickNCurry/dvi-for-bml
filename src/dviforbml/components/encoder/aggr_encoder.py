from enum import Enum
from typing import Tuple

from torch import Tensor

from dviforbml.components.encoder.abstract_encoder import AbstractEncoder


class Aggr(Enum):
    MEAN = "mean"
    MAX = "max"


class AggrEncoder(AbstractEncoder):
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
        aggregation: Aggr,
    ) -> None:
        super(AggrEncoder, self).__init__(
            c_dim,
            h_dim,
            z_dim,
            num_layers,
            non_linearity,
            num_heads,
            num_blocks,
            max_context_size,
        )

        self.aggregation = aggregation

    def forward(self, context: Tensor, mask: Tensor | None) -> Tuple[Tensor, Tensor]:
        # (batch_size, num_subtasks, data_size, c_dim)
        # (batch_size, num_subtasks, data_size)

        h = self.compute_h(context, mask)
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

        s_emb = self.compute_s_emb(context, mask)
        # (batch_size, num_subtasks, z_dim)

        return r, s_emb
