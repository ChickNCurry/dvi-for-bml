import torch
from torch import Tensor, nn


class HyperNet(nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        non_linearity: str,
        num_steps: int,
    ) -> None:
        super(HyperNet, self).__init__()

        self.proj_t = nn.Embedding(num_steps + 1, h_dim)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
        )

    def forward(self, t: int, context_embedding: Tensor) -> Tensor:
        # (1), (batch_size, h_dim)

        t = self.proj_t(torch.tensor([t], device=context_embedding.device))
        # (batch_size, h_dim)

        hyper: Tensor = self.mlp(t + context_embedding)
        # (batch_size, h_dim)

        return hyper
