import torch
from torch import Tensor, nn


class ControlFunction(nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
        num_steps: int,
        set_encoder: nn.Module,
    ) -> None:
        super(ControlFunction, self).__init__()

        self.proj_t = nn.Embedding(num_steps + 1, h_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)
        self.proj_c = set_encoder

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            *[
                layer
                for layer in (
                    getattr(nn, non_linearity)(),
                    nn.Linear(h_dim, h_dim),
                )
                for _ in range(num_layers - 1)
            ]
        )

        self.proj_out = nn.Linear(h_dim, z_dim)

    def forward(self, z: Tensor, t: int, c: Tensor) -> Tensor:
        # (batch_size, z_dim), (1), (batch_size, context_size, c_dim)

        z = self.proj_z(z)
        t = self.proj_t(torch.tensor([t], device=z.device))
        c = self.proj_c(c)
        # (batch_size, h_dim)

        control: Tensor = self.proj_out(self.mlp(z + t + c))
        # (batch_size, z_dim)

        return control


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

    def forward(self, context: Tensor) -> Tensor:
        # (batch_size, context_size, c_dim)

        h: Tensor = context

        h = self.mlp(h)
        # (batch_size, context_size, h_dim)

        if self.is_attentive:
            h, _ = self.self_attn(h, h, h, need_weights=False)
            # (batch_size, context_size, h_dim)

        h = self.aggregation(h, dim=1)
        # (batch_size, h_dim)

        h = h + self.context_size_embedding(
            torch.tensor([context.shape[1]], device=h.device)
        )
        # (batch_size, h_dim)

        return h


class TestEncoder(nn.Module):
    def __init__(self, c_dim: int, h_dim: int) -> None:
        super(TestEncoder, self).__init__()

        self.proj_c = nn.Linear(c_dim, h_dim)

    def forward(self, c: Tensor) -> Tensor:

        c = self.proj_c(c.squeeze(1))

        return c
