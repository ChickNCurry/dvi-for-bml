import torch
from torch import Tensor

from dviforbml.components.schedule.abstract_schedule import AbstractSchedule


class StepSizeSchedule(AbstractSchedule):
    def __init__(self, num_steps: int, device: torch.device) -> None:
        super(StepSizeSchedule, self).__init__()

        self.num_entries = num_steps + 1
        self.step_size = torch.tensor(1 / self.num_entries, device=device)

    def get(self, n: int) -> Tensor:
        return self.step_size
