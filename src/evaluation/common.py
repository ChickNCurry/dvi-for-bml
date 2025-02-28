from dataclasses import dataclass
from enum import Enum
from typing import Any

from torch.utils.data import DataLoader

from src.architectures.np import NP


class ModelType(Enum):
    LNP = "lnp"
    CNP = "cnp"
    DVINP = "dvinp"


@dataclass
class ModelInfo:
    name: str
    project: str
    type: ModelType
    model: NP | None = None
    val_loader: DataLoader[Any] | None = None
