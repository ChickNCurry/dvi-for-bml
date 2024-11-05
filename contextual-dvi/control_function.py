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
    ) -> None:
        super(ControlFunction, self).__init__()

        self.proj_t = nn.Embedding(num_steps + 1, h_dim)
        self.proj_z = nn.Linear(z_dim, h_dim)

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

    def forward(self, z: Tensor, t: int, context_embedding: Tensor) -> Tensor:
        # (batch_size, z_dim), (1), (batch_size, h_dim)

        z = self.proj_z(z)
        t = self.proj_t(torch.tensor([t], device=z.device))
        # (batch_size, h_dim)

        control: Tensor = self.proj_out(self.mlp(z + t + context_embedding))
        # (batch_size, z_dim)

        return control
