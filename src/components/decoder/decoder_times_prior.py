import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from src.components.decoder.decoder import Decoder


class DecoderTimesPrior(Distribution):
    def __init__(
        self,
        decoder: Decoder,
        x_target: Tensor,
        y_target: Tensor,
        mask: Tensor | None,
    ) -> None:
        # (batch_size, num_subtasks, z_dim)
        # (batch_size, num_subtasks, target_size, x_dim)
        # (batch_size, num_subtasks, target_size, y_dim)
        # (batch_size, num_subtasks, context_size)

        super(DecoderTimesPrior, self).__init__(validate_args=False)

        self.decoder = decoder
        self.x_target = x_target
        self.y_target = y_target
        self.mask = mask

        batch_size = x_target.shape[0]
        num_subtasks = x_target.shape[1]
        z_dim = decoder.z_dim
        device = x_target.device

        self.prior = Normal(  # type: ignore
            torch.zeros((batch_size, num_subtasks, z_dim), device=device),
            torch.ones((batch_size, num_subtasks, z_dim), device=device),
        )

    def log_prob(self, z: Tensor) -> Tensor:
        # (batch_size, num_subtasks, z_dim)

        log_like: Tensor = self.decoder(z, self.x_target).log_prob(self.y_target)
        # (batch_size, num_subtasks, target_size, y_dim)

        if self.mask is not None:
            mask = self.mask.unsqueeze(-1).expand(-1, -1, -1, log_like.shape[-1])
            # (batch_size, num_subtasks, target_size, y_dim)

            log_like = log_like * mask

        log_like = log_like.sum(dim=2).sum(dim=-1, keepdim=True)
        # (batch_size, num_subtasks, 1)

        log_prior: Tensor = self.prior.log_prob(z)  # type: ignore
        log_prior = log_prior.sum(dim=2, keepdim=True)
        # (batch_size, num_subtasks, 1)

        return log_like + log_prior
