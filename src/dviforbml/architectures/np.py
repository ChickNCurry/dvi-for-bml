from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

from dviforbml.components.decoder.decoder import Decoder
from dviforbml.components.encoder.abstract_encoder import AbstractEncoder


class NP(nn.Module, ABC):
    def __init__(self, encoder: AbstractEncoder, decoder: Decoder) -> None:
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

    def freeze(self) -> None:
        self.freeze_encoder()
        self.freeze_decoder()

    def unfreeze(self) -> None:
        self.unfreeze_encoder()
        self.unfreeze_decoder()

    def freeze_decoder(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad = True

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True
