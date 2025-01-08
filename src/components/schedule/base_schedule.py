from abc import ABC, abstractmethod

from torch import Tensor


class BaseSchedule(ABC):
    def update(self, r: Tensor) -> None:
        pass

    @abstractmethod
    def get(self, n: int) -> Tensor:
        pass
