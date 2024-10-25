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
        self.set_encode_c = SetEncoder(
            c_dim, h_dim, num_layers, non_linearity, False, "mean"
        )  # nn.Linear(c_dim, h_dim)

        self.blocks = nn.ModuleList(
            [ResidualBlock(h_dim, non_linearity) for _ in range(num_layers)]
        )

        self.proj_score = nn.Linear(h_dim, z_dim)

    def forward(self, z: Tensor, t: int, c: Tensor) -> Tensor:
        # (batch_size, z_dim), (1), (batch_size, context_size, c_dim)

        z = self.proj_z(z)
        t = self.proj_t(torch.tensor([t], device=z.device))
        c = self.set_encode_c(c)
        # (batch_size, h_dim)

        for block in self.blocks:
            score: Tensor = block(z, t, c)
            # (batch_size, h_dim)

        score = self.proj_score(score)
        # (batch_size, z_dim)

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
        # (batch_size, h_dim), (batch_size, h_dim), (batch_size, h_dim)

        out: Tensor = self.mlp_out(self.mlp_z(z) + self.mlp_t(t) + self.mlp_c(c)) + z
        # (batch_size, h_dim)

        return out


class SetEncoder(nn.Module):
    def __init__(
        self,
        c_dim: int,
        h_dim: int,
        num_layers: int,
        non_linearity: str,
        is_attentive: bool,
        aggregation: str,
    ) -> None:
        super(SetEncoder, self).__init__()

        self.aggregation = getattr(torch, aggregation) if aggregation else None

        self.is_attentive = is_attentive

        self.mlp = nn.Sequential(
            nn.Linear(c_dim, h_dim - 1),
            *[
                layer
                for layer in (
                    getattr(nn, non_linearity)(),
                    nn.Linear(h_dim - 1, h_dim - 1),
                )
                for _ in range(num_layers - 1)
            ]
        )

        if self.is_attentive:
            self.self_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

    def forward(self, context: Tensor) -> Tensor:
        # (batch_size, context_size, c_dim)

        h: Tensor = context

        if self.is_attentive:
            h, _ = self.self_attn(h, h, h, need_weights=False)
            # (batch_size, context_size, h_dim - 1)
        else:
            h = self.mlp(h)
            # (batch_size, context_size, h_dim - 1)

        h = self.aggregation(h, dim=1)
        # (batch_size, h_dim - 1)

        context_size_feature = (
            torch.tensor([context.shape[1]], device=context.device)
            .unsqueeze(0)
            .repeat(context.shape[0], 1)
        )
        # (batch_size, 1)

        h = torch.cat([h, context_size_feature], dim=1)
        # (batch_size, h_dim)

        return h
