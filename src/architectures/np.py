from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

from src.components.decoder.decoder import Decoder
from src.components.encoder.base_encoder import BaseEncoder


class NP(nn.Module, ABC):
    def __init__(self, encoder: BaseEncoder, decoder: Decoder) -> None:
        super(NP, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    @abstractmethod
    def inference(
        self, x_context: Tensor, y_context: Tensor, mask: Tensor | None, x_data: Tensor
    ) -> Tuple[Normal, Tensor | None]:
        raise NotImplementedError

    @abstractmethod
    def get_target_dist(
        self, x_context: Tensor, y_context: Tensor, mask: Tensor | None
    ) -> Distribution | None:
        raise NotImplementedError
