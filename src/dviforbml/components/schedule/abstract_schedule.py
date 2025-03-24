from abc import ABC, abstractmethod

from torch import Tensor, nn


class AbstractSchedule(ABC, nn.Module):
    def update(self, r: Tensor, mask: Tensor | None, s: Tensor | None) -> None:
        pass

    @abstractmethod
    def get(self, n: int) -> Tensor:
        pass
