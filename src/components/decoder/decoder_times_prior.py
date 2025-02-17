import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from src.components.decoder.decoder import Decoder


class DecoderTimesPrior(Distribution):
    def __init__(
        self,
        decoder: Decoder,
        x: Tensor,
        y: Tensor,
        mask: Tensor | None,
    ) -> None:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, data_size, x_dim)
        # (batch_size, num_subtasks, data_size, y_dim)
        # (batch_size, num_subtasks, data_size)

        super(DecoderTimesPrior, self).__init__(validate_args=False)

        self.decoder = decoder
        self.x = x
        self.y = y
        self.mask = mask

        batch_size = x.shape[0]
        num_subtasks = x.shape[1]
        device = x.device
        z_dim = decoder.z_dim

        self.prior = Normal(  # type: ignore
            torch.zeros((batch_size, num_subtasks, z_dim), device=device),
            torch.ones((batch_size, num_subtasks, z_dim), device=device),
        )

    def log_prob(self, z: Tensor) -> Tensor:
        # (batch_size, num_subtasks, z_dim)

        log_like: Tensor = self.decoder(z, self.x).log_prob(self.y)
        # (batch_size, num_subtasks, data_size, y_dim)

        if self.mask is not None:
            mask = self.mask.unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
            log_like = log_like * mask
            # (batch_size, num_subtasks, data_size, y_dim)

        log_like = log_like.sum(dim=2).sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        log_prior: Tensor = self.prior.log_prob(z)  # type: ignore
        log_prior = log_prior.sum(dim=2, keepdim=True)
        # (batch_size, num_subtasks, 1)

        return log_like + log_prior
