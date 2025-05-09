from typing import Tuple
from torch import Tensor, nn


class ProjEncoder(nn.Module):
    def __init__(self, c_dim: int, h_dim: int) -> None:
        super(ProjEncoder, self).__init__()

        self.proj_c = nn.Linear(c_dim, h_dim)

    def forward(self, context: Tensor, mask: Tensor | None) -> Tuple[Tensor, Tensor]:
        # (batch_size, num_subtasks, context_size, c_dim)

        assert context.shape[2] == 1

        context = context.squeeze(2)
        # (batch_size, num_subtasks, c_dim)

        r: Tensor = self.proj_c(context)
        # (batch_size, num_subtasks, h_dim)

        return r, None
