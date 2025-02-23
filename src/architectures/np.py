from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor, nn
from torch.distributions.normal import Normal


class NP(nn.Module, ABC):
    def __init__(self) -> None:
        super(NP, self).__init__()

    @abstractmethod
    def inference(
        self, x_context: Tensor, y_context: Tensor, mask: Tensor | None, x_data: Tensor
    ) -> Tuple[Normal, Tensor | None]:
        raise NotImplementedError
