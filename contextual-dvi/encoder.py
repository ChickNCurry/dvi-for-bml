import torch
from torch import Tensor, nn


class SetEncoder(nn.Module):
    def __init__(
        self,
        c_dim: int,
        h_dim: int,
        num_layers: int,
        non_linearity: str,
        is_attentive: bool,
        aggregation: str,
        max_context_size: int,
    ) -> None:
        super(SetEncoder, self).__init__()

        # self.context_embedding_dim = int(h_dim / 2)
        # self.context_size_embedding_dim = h_dim - self.context_embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(c_dim, h_dim),
            *[
                layer
                for layer in (
                    getattr(nn, non_linearity)(),
                    nn.Linear(h_dim, h_dim),
                )
                for _ in range(num_layers - 1)
            ]
        )

        self.is_attentive = is_attentive

        if self.is_attentive:
            self.self_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

        self.aggregation = getattr(torch, aggregation)

        self.context_size_embedding = nn.Embedding(max_context_size + 1, h_dim)

    def forward(self, context: Tensor, mask: Tensor | None = None) -> Tensor:
        # (batch_size, context_size, c_dim), (batch_size, context_size)

        c: Tensor = context

        c = self.mlp(c)
        # (batch_size, context_size, h_dim)

        if self.is_attentive:
            c, _ = self.self_attn(c, c, c, need_weights=False)
            # (batch_size, context_size, h_dim)

        if mask is not None:
            c = c * mask.unsqueeze(-1)

            counts = mask.sum(dim=1, keepdim=True)

            c = c.sum(dim=1) / counts
            # (batch_size, h_dim)

            e: Tensor = self.context_size_embedding(counts.squeeze(1).int())
            # (batch_size, h_dim)

            # out = torch.cat([c, e], dim=1)
            out = c + e
            # (batch_size, h_dim)
        else:
            c = self.aggregation(c, dim=1)
            # (batch_size, h_dim)

            e = self.context_size_embedding(
                torch.tensor([context.shape[1]], device=c.device)
            ).expand(c.shape[0], -1)
            # (batch_size, h_dim)

            # out = torch.cat([c, e], dim=1)
            out = c + e
            # (batch_size, h_dim)

        return out


class TestEncoder(nn.Module):
    def __init__(self, c_dim: int, h_dim: int) -> None:
        super(TestEncoder, self).__init__()

        self.proj_c = nn.Linear(c_dim, h_dim)

    def forward(self, c: Tensor, mask: Tensor | None = None) -> Tensor:

        c = self.proj_c(c.squeeze(1))

        return c
