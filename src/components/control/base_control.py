from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor, nn


class BaseControl(nn.Module, ABC):
    def __init__(self) -> None:
        super(BaseControl, self).__init__()

    @abstractmethod
    def forward(
        self,
        n: int,
        z: Tensor,
        r: Tensor | Tuple[Tensor, Tensor],
        mask: Tensor | None,
        score: Tensor | None,
    ) -> Tensor:
        pass
