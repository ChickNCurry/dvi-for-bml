from typing import Callable

from torch import Tensor, nn
from torch.distributions import Distribution

from src.components.cdvi.cdvi import CDVI
from src.components.encoder.abstract_encoder import AbstractEncoder


class DVI(nn.Module):
    def __init__(
        self,
        encoder: AbstractEncoder,
        cdvi: CDVI,
        contextual_target: Callable[[Tensor, Tensor | None], Distribution] | None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.cdvi = cdvi
        self.contextual_target = contextual_target
