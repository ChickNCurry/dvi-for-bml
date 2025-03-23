from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor, nn


class AbstractEncoder(nn.Module, ABC):
    def __init__(
        self,
    ) -> None:
        super(AbstractEncoder, self).__init__()

    @abstractmethod
    def forward(
        self, context: Tensor, mask: Tensor | None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        pass
