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


class StepSizeSchedule(Schedule):
    def __init__(self, num_steps: int, device: torch.device) -> None:
        self.step_size = torch.tensor(1 / num_steps, device=device)

    def update(self, r: Tensor) -> None:
        pass

    def get(self, n: int) -> Tensor:
        return self.step_size


class CosineStepSizeSchedule(Schedule, nn.Module):
    def __init__(
        self,
        num_steps: int,
        device: torch.device,
        init_val: float = 0.1,
        require_grad: bool = True,
    ) -> None:
        super(CosineStepSizeSchedule, self).__init__()

        self.amplitude = nn.Parameter(
            torch.tensor([1], device=device) * init_val, requires_grad=require_grad
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


class ContextualCosineStepSizeSchedule(Schedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        init_val: float = 0.1,
    ) -> None:
        super(ContextualCosineStepSizeSchedule, self).__init__()

        self.amplitude_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, 1),
            nn.Softplus(),
        )

        # init amplitude_mlp such that amplitude is init_val
        nn.init.constant_(self.amplitude_mlp[0].weight, 0)
        nn.init.constant_(self.amplitude_mlp[0].bias, 0)
        nn.init.constant_(self.amplitude_mlp[2].weight, 0)
        nn.init.constant_(self.amplitude_mlp[2].bias, init_val)

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


class NoiseSchedule(Schedule):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
        min: float = 0.01,
        max: float = 1,
        requires_grad: bool = True,
    ) -> None:

        self.noise_schedule = nn.Parameter(
            torch.linspace(max, min, num_steps, device=device)
            .unsqueeze(1)
            .expand(num_steps, z_dim),
            requires_grad=requires_grad,
        )

    def update(self, r: Tensor) -> None:
        pass

    def get(self, n: int) -> Tensor:
        var_n = torch.nn.functional.softplus(self.noise_schedule[n - 1, :])
        return var_n


class ContextualNoiseSchedule(Schedule, nn.Module):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        device: torch.device,
        min: float = 0.01,
        max: float = 1,
    ) -> None:
        super(ContextualNoiseSchedule, self).__init__()

        self.z_dim = z_dim
        self.num_steps = num_steps

        self.noise_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, num_steps * z_dim),
        )

        nn.init.constant_(self.noise_mlp[-1].weight, 0)
        nn.init.constant_(self.noise_mlp[-1].bias, 0)

        self.noise_init = torch.linspace(max, min, num_steps, device=device)

    def update(self, r: Tensor) -> None:
        noise: Tensor = self.noise_mlp(r)
        noise = noise.view(r.shape[0], r.shape[1], self.z_dim, self.num_steps)

        noise_init = self.noise_init[None, None, None, :]

        self.noise_schedule = nn.functional.softplus(noise + noise_init)

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        var_n: Tensor = self.noise_schedule[:, :, :, n - 1]
        # start at amplitude
        # never reaches 0

        return var_n


class CosineNoiseSchedule(Schedule, nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_steps: int,
        device: torch.device,
        init_val: float = 1.0,
        requires_grad: bool = True,
    ) -> None:
        super(CosineNoiseSchedule, self).__init__()

        self.amplitude = nn.Parameter(
            torch.ones((z_dim), device=device) * init_val,
            requires_grad=requires_grad,
        )

        self.cosine_square_schedule: List[float] = [
            np.square(np.cos((math.pi / 2) * (n / num_steps)))
            for n in range(0, num_steps)
        ]

    def update(self, r: Tensor) -> None:
        pass

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        var_n = (
            nn.functional.softplus(self.amplitude) * self.cosine_square_schedule[n - 1]
        )
        # start at amplitude
        # never reaches 0

        return var_n


class ContextualCosineNoiseSchedule(Schedule, nn.Module):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        init_val: float = 1.0,
    ) -> None:
        super(ContextualCosineNoiseSchedule, self).__init__()

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
        nn.init.constant_(self.amplitude_mlp[2].bias, init_val)

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


class AnnealingSchedule(Schedule, nn.Module):
    def __init__(
        self,
        num_steps: int,
        device: torch.device,
        requires_grad: bool = True,
    ) -> None:
        super(AnnealingSchedule, self).__init__()

        # init params to same val such that betas increase linearly
        self.params = nn.Parameter(
            torch.ones((num_steps), device=device),
            requires_grad=requires_grad,
        )

    def update(self, r: Tensor) -> None:
        params = torch.nn.functional.softplus(self.params)
        self.betas = torch.cumsum(params, dim=0) / torch.sum(params, dim=0)

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        beta_n = self.betas[n - 1]
        # starts higher than 0
        # reaches 1

        return beta_n


class ContextualAnnealingSchedule(Schedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
        device: torch.device,
    ) -> None:
        super(ContextualAnnealingSchedule, self).__init__()

        self.num_steps = num_steps

        self.params_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, num_steps),
        )

        nn.init.constant_(self.params_mlp[-1].weight, 0)
        nn.init.constant_(self.params_mlp[-1].bias, 0)

        self.params_init = torch.ones((num_steps), device=device) / num_steps

    def update(self, r: Tensor) -> None:
        self.params: Tensor = nn.functional.softplus(
            self.params_mlp(r) + self.params_init[None, None, :]
        )

        self.params = self.params / torch.sum(self.params, dim=-1, keepdim=True)
        self.betas = torch.cumsum(self.params, dim=-1)

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        beta_n = self.betas[:, :, n - 1][:, :, None]
        # starts higher than 0
        # reaches 1

        return beta_n


class ContextualCosineAnnealingSchedule(Schedule, nn.Module):
    def __init__(
        self,
        h_dim: int,
        non_linearity: str,
        num_steps: int,
    ) -> None:
        super(ContextualCosineAnnealingSchedule, self).__init__()

        self.params_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            getattr(nn, non_linearity)(),
            nn.Linear(h_dim, num_steps),
            nn.Softplus(),
        )

        # init params_mlp corresponding to linearly increasing betas
        nn.init.constant_(self.params_mlp[0].weight, 0)
        nn.init.constant_(self.params_mlp[0].bias, 0)
        nn.init.constant_(self.params_mlp[2].weight, 0)
        nn.init.constant_(self.params_mlp[2].bias, 1 / num_steps)

        self.num_steps = num_steps

    def update(self, r: Tensor) -> None:
        self.params: Tensor = self.params_mlp(r)
        self.params = self.params / torch.sum(self.params, dim=-1, keepdim=True)
        self.betas = torch.cumsum(self.params, dim=-1)

    def get(self, n: int) -> Tensor:
        # transition from n-1 to n

        beta_n = self.betas[:, :, n - 1]
        # starts higher than 0
        # reaches 1

        return beta_n
