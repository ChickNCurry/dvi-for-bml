from abc import ABC, abstractmethod
from typing import Any, Tuple

from torch import Tensor, nn


class BaseControl(nn.Module, ABC):
    def __init__(
        self,
    ) -> None:
        super(BaseControl, self).__init__()

    @abstractmethod
    def forward(self, n: int, z: Tensor, rest: Tuple[Any]) -> Tensor:
        pass
