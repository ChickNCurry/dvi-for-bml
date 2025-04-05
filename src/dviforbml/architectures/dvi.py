from typing import Callable, Tuple

from torch import Tensor, nn
from torch.distributions import Distribution

from dviforbml.components.cdvi.cdvi import CDVI
from dviforbml.components.encoder.abstract_encoder import AbstractEncoder
from torch.distributions.normal import Normal


class DVI(nn.Module):
    def __init__(
        self,
        encoder: AbstractEncoder,
        cdvi: CDVI,
        contextual_target: Callable[[Tensor, Tensor | None], Distribution],
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.cdvi = cdvi
        self.contextual_target = contextual_target

    def inference(
        self, context: Tensor, mask: Tensor | None, x_data: Tensor
    ) -> Tuple[Normal, Tensor | None]:
        r_context, s_context = self.encoder(context, mask)

        target_dist = self.get_target_dist(context, mask)

        _, zs = self.cdvi.run_forward_process(
            target_dist, r_context, mask, s_context, None
        )

        assert zs is not None

        y_dist_data: Normal = self.decoder(zs[-1], x_data)

        return y_dist_data, zs[-1]

    def get_target_dist(self, context: Tensor, mask: Tensor | None):
        return self.contextual_target(context, mask)
