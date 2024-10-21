import torch
from torch import Tensor, nn


class ScoreFunction(nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
        num_steps: int,
        c_dim: int,
    ) -> None:
        super(ScoreFunction, self).__init__()

        self.proj_t = nn.Embedding(num_steps + 1, h_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)
        self.proj_c = nn.Linear(c_dim, h_dim)

        self.blocks = nn.ModuleList(
            [ResidualBlock(h_dim, non_linearity) for _ in range(num_layers)]
        )

        self.proj_score = nn.Linear(h_dim, z_dim)

    def forward(self, z: Tensor, t: int, c: Tensor) -> Tensor:

        z = self.proj_z(z)
        t = self.proj_t(torch.tensor([t], device=z.device))
        c = self.proj_c(c)

        for block in self.blocks:
            score: Tensor = block(z, t, c)

        score = self.proj_score(score)

        return score


class ResidualBlock(nn.Module):
    def __init__(self, h_dim: int, non_linearity: str) -> None:
        super(ResidualBlock, self).__init__()

        self.mlp_t = nn.Linear(h_dim, 2 * h_dim)

        self.mlp_z = nn.Sequential(
            nn.LayerNorm(h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, 2 * h_dim),
        )

        self.mlp_c = nn.Sequential(
            nn.LayerNorm(h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, 2 * h_dim),
        )

        self.mlp_out = nn.Sequential(
            getattr(nn, non_linearity)(),
            nn.Linear(2 * h_dim, h_dim),
            getattr(nn, non_linearity)(),
        )

    def forward(self, z: Tensor, t: Tensor, c: Tensor) -> Tensor:
        out: Tensor = self.mlp_out(self.mlp_z(z) + self.mlp_t(t) + self.mlp_c(c)) + z

        return out
