import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch import Tensor, nn


class Schedule(ABC):
    def update(self, r: Tensor) -> None:
        pass

    @abstractmethod
    def get(self, n: int) -> Tensor:
        pass


class ConstantStepSizeSchedule(Schedule):
    def __init__(self, z_dim: int, num_steps: int, device: torch.device) -> None:
        self.step_size = torch.tensor(1 / num_steps, device=device)

    def update(self, r: Tensor) -> None:
        pass

    def get(self, n: int) -> Tensor:
        return self.step_size


class CosineStepSizeSchedule(Schedule, nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
        require_grad: bool = True,
    ) -> None:
        super(CosineStepSizeSchedule, self).__init__()

        self.amplitude = nn.Parameter(
            torch.ones((z_dim), device=device) * 0.1, requires_grad=require_grad
        )

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ][::-1]

    def update(self, r: Tensor) -> None:
        pass

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        delta_t_n = (
            torch.nn.functional.softplus(self.amplitude)
            * self.cosine_square_schedule[n - 1]
        )
        # starts higher than 0
        # reaches amplitude

        return delta_t_n


class ContextualStepSizeSchedule(Schedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        non_linearity: str,
        num_steps: int,
    ) -> None:
        super(ContextualStepSizeSchedule, self).__init__()

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        # init amplitude_mlp such that amplitude is 0.1
        nn.init.constant_(self.amplitude_mlp[0].weight, 0)
        nn.init.constant_(self.amplitude_mlp[0].bias, 0)
        nn.init.constant_(self.amplitude_mlp[2].weight, 0)
        nn.init.constant_(self.amplitude_mlp[2].bias, 0.1)

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ][::-1]

    def update(self, r: Tensor) -> None:
        self.amplitude = self.amplitude_mlp(r)

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        delta_t_n: Tensor = self.amplitude * self.cosine_square_schedule[n - 1]
        # starts higher than 0
        # reaches amplitude

        return delta_t_n


class CosineNoiseSchedule(Schedule, nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
        requires_grad: bool = True,
    ) -> None:
        super(CosineNoiseSchedule, self).__init__()

        self.amplitude = nn.Parameter(
            torch.ones((z_dim), device=device) * 1.0, requires_grad=requires_grad
        )

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ]

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        var_n = (
            nn.functional.softplus(self.amplitude) * self.cosine_square_schedule[n - 1]
        )
        # start at amplitude
        # never reaches 0

        return var_n


class ContextualNoiseSchedule(Schedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        non_linearity: str,
        num_steps: int,
    ) -> None:
        super(ContextualNoiseSchedule, self).__init__()

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        # init amplitude_mlp such that amplitude is 1
        nn.init.constant_(self.amplitude_mlp[0].weight, 0)
        nn.init.constant_(self.amplitude_mlp[0].bias, 0)
        nn.init.constant_(self.amplitude_mlp[2].weight, 0)
        nn.init.constant_(self.amplitude_mlp[2].bias, 1)

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ]

    def update(self, r: Tensor) -> None:
        self.amplitude = self.amplitude_mlp(r)

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        var_n: Tensor = self.amplitude * self.cosine_square_schedule[n - 1]
        # start at amplitude
        # never reaches 0

        return var_n


class ConstantAnnealingSchedule(Schedule, nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
        requires_grad: bool = True,
    ) -> None:
        super(ConstantAnnealingSchedule, self).__init__()

        # init params such that betas increase linearly
        self.params = nn.Parameter(
            torch.ones((num_steps, z_dim), device=device) / num_steps,
            requires_grad=requires_grad,
        )

    def update(self, r: Tensor) -> None:
        pass

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        params = torch.nn.functional.softplus(self.params)
        betas = torch.cumsum(params, dim=0) / torch.sum(params, dim=0)
        beta_n = betas[n - 1]
        # starts higher than 0
        # reaches 1

        return beta_n


class ContextualAnnealingSchedule(Schedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        non_linearity: str,
        num_steps: int,
    ) -> None:
        super(ContextualAnnealingSchedule, self).__init__()

        self.params_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, num_steps * z_dim),
            nn.Softplus(),
        )

        # init params_mlp corresponding to linearly increasing betas
        nn.init.constant_(self.params_mlp[0].weight, 0)
        nn.init.constant_(self.params_mlp[0].bias, 0)
        nn.init.constant_(self.params_mlp[2].weight, 0)
        nn.init.constant_(self.params_mlp[2].bias, 1 / num_steps)

        self.num_steps = num_steps
        self.z_dim = z_dim

    def update(self, r: Tensor) -> None:
        self.params: Tensor = self.params_mlp(r)
        self.params = self.params.view(
            r.shape[0], r.shape[1], self.z_dim, self.num_steps
        )
        self.params = self.params / torch.sum(self.params, dim=-1, keepdim=True)

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        betas = torch.cumsum(self.params, dim=-1)
        beta_n = betas[:, :, :, n - 1]
        # starts higher than 0
        # reaches 1

        return beta_n
