import torch
from torch import Tensor, nn


class ScoreFunction(nn.Module):
    def __init__(
        self,
        c_dim: int,
        h_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
        num_steps: int,
    ) -> None:
        super(ScoreFunction, self).__init__()

        self.time_embedding = nn.Embedding(num_steps, z_dim)

        self.score_mlp = nn.Sequential(
            nn.Linear(z_dim + c_dim, h_dim),
            *[
                layer
                for layer in (
                    getattr(nn, non_linearity)(),
                    nn.Linear(h_dim, h_dim),
                )
                for _ in range(num_layers - 1)
            ],
            nn.Linear(h_dim, z_dim),
        )

    def forward(self, z: Tensor, t: int, c: Tensor) -> Tensor:

        time = self.time_embedding(torch.tensor([t], device=z.device))
        # (batch_size, z_dim)

        z = z + time

        score = torch.cat([z, c], dim=1)
        score = self.score_mlp(score)
        # (batch_size, z_dim)

        return score
