from abc import ABC, abstractmethod

from torch import Tensor


class BaseSchedule(ABC):
    def update(self, r: Tensor, mask: Tensor | None) -> None:
        pass

    @abstractmethod
    def get(self, n: int) -> Tensor:
        pass
