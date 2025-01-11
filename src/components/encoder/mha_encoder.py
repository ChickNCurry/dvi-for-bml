from torch import Tensor, nn

from src.components.encoder.base_encoder import BaseEncoder


class MHAEncoder(BaseEncoder):
    def __init__(
        self,
        c_dim: int,
        h_dim: int,
        num_layers: int,
        non_linearity: str,
        num_heads: int | None,
    ) -> None:
        super(MHAEncoder, self).__init__()

        self.num_heads = num_heads

        self.proj_in = nn.Linear(c_dim, h_dim)

        if self.num_heads is not None:
            self.self_attn = nn.MultiheadAttention(h_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 1)
            ],
        )

    def forward(self, context: Tensor, mask: Tensor | None) -> Tensor:
        # (batch_size, num_subtasks, context_size, c_dim)
        # (batch_size, num_subtasks, context_size)

        batch_size = context.shape[0]
        num_subtasks = context.shape[1]
        context_size = context.shape[2]

        h: Tensor = self.proj_in(context)
        # (batch_size, num_subtasks, context_size, h_dim)

        if self.num_heads is not None:
            h = h.view(batch_size * num_subtasks, context_size, -1)
            # (batch_size * num_subtasks, context_size, h_dim)

            key_padding_mask = (
                (mask.view(batch_size * num_subtasks, -1).bool().logical_not())
                if mask is not None
                else None
            )  # (batch_size * num_subtasks, context_size)

            h, _ = self.self_attn(
                query=h,
                key=h,
                value=h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )  # (batch_size * num_subtasks, context_size, h_dim)

            h = h.view(batch_size, num_subtasks, context_size, -1)
            # (batch_size, num_subtasks, context_size, h_dim)

        h = self.mlp(h)
        # (batch_size, num_subtasks, context_size, h_dim)

        r = h if mask is None else h * mask.unsqueeze(-1)
        # (batch_size, num_subtasks, context_size, h_dim)

        return r
