from dataclasses import dataclass

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from src.architectures.dvi import DVI
from src.components.cdvi.cmcd import CMCD
from src.components.cdvi.dis import DIS
from src.components.control.aggr_control import AggrControl
from src.components.control.bca_control import BCAControl
from src.components.control.mha_control import MHAControl
from src.components.encoder.aggr_encoder import Aggr, AggrEncoder
from src.components.encoder.bca_encoder import BCAEncoder
from src.components.encoder.mha_encoder import MHAEncoder
from src.components.schedule.annealing_schedule import (
    AggrAnnealingSchedule,
    AnnealingSchedule,
    BCAAnnealingSchedule,
)
from src.components.schedule.cos_noise_schedule import (
    AggrCosineNoiseSchedule,
    BCACosineNoiseSchedule,
    CosineNoiseSchedule,
)
from src.components.schedule.noise_schedule import (
    AggrNoiseSchedule,
    BCANoiseSchedule,
    NoiseSchedule,
)
from src.components.schedule.step_size_schedule import StepSizeSchedule
from src.evaluation.visualization.visualize_dvi import visualize_dvi_2d
from src.training.dvi_trainer import DVITrainerContext
from src.utils.datasets import ContextSetDataset
from src.utils.distros import TaskPosteriorGMM


def main() -> None:
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     try:
    #         import torch_directml  # type: ignore

    #         device = torch_directml.device()
    #     except ImportError:
    #         device = torch.device("cpu")

    device = torch.device("cpu")

    @dataclass
    class Config:
        num_steps = 16
        c_dim = 2
        z_dim = 2
        h_dim = 32
        num_layers = 3
        non_linearity = "GELU"
        use_score = False
        learning_rate = 3e-3
        size = 64
        batch_size = 64

    config = Config()

    dataset = ContextSetDataset(
        size=config.size, c_dim=config.c_dim, max_context_size=10, sampling_factor=4
    )
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)

    contextual_target = TaskPosteriorGMM

    aggr_encoder = AggrEncoder(
        c_dim=config.c_dim,
        h_dim=config.h_dim,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        num_heads=None,
        aggregation=Aggr.MEAN,
        max_context_size=None,  # dataset.max_context_size
    )

    bca_encoder = BCAEncoder(
        c_dim=config.c_dim,
        h_dim=config.h_dim,
        z_dim=config.z_dim,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        num_heads=None,
    )

    mha_encoder = MHAEncoder(
        c_dim=config.c_dim,
        h_dim=config.h_dim,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        num_heads=1,
    )

    aggr_control = AggrControl(
        h_dim=config.h_dim,
        z_dim=config.z_dim,
        num_steps=config.num_steps,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        use_score=config.use_score,
        use_error=False,
    )

    bca_control = BCAControl(
        h_dim=config.h_dim,
        z_dim=config.z_dim,
        num_steps=config.num_steps,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        use_score=config.use_score,
        use_error=False,
    )

    mha_control = MHAControl(
        h_dim=config.h_dim,
        z_dim=config.z_dim,
        num_steps=config.num_steps,
        num_layers=config.num_layers,
        non_linearity=config.non_linearity,
        use_score=config.use_score,
        num_heads=1,
    )

    step_size_schedule = StepSizeSchedule(num_steps=config.num_steps, device=device)

    noise_schedule = NoiseSchedule(
        z_dim=config.z_dim,
        num_steps=config.num_steps,
        device=device,
    )

    aggr_noise_schedule = AggrNoiseSchedule(
        z_dim=config.z_dim,
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
        device=device,
    )

    bca_noise_schedule = BCANoiseSchedule(
        z_dim=config.z_dim,
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
        device=device,
    )

    # mha_noise_schedule = MHANoiseSchedule(
    #     z_dim=config.z_dim,
    #     h_dim=config.h_dim,
    #     non_linearity=config.non_linearity,
    #     num_steps=config.num_steps,
    #     device=device,
    #     num_heads=1
    # )

    annealing_schedule = AnnealingSchedule(num_steps=config.num_steps, device=device)

    aggr_annealing_schedule = AggrAnnealingSchedule(
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
        device=device,
    )

    bca_annealing_schedule = BCAAnnealingSchedule(
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
        device=device,
    )

    # mha_annealing_schedule = MHAAnnealingSchedule(
    #     h_dim=config.h_dim,
    #     non_linearity=config.non_linearity,
    #     num_steps=config.num_steps,
    #     device=device,
    #     num_heads=1
    # )

    cos_noise_schedule = CosineNoiseSchedule(
        z_dim=config.z_dim,
        num_steps=config.num_steps,
        device=device,
    )

    aggr_cos_noise_schedule = AggrCosineNoiseSchedule(
        z_dim=config.z_dim,
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
    )

    bca_cos_noise_schedule = BCACosineNoiseSchedule(
        z_dim=config.z_dim,
        h_dim=config.h_dim,
        non_linearity=config.non_linearity,
        num_steps=config.num_steps,
    )

    comps = (aggr_control, noise_schedule, annealing_schedule, aggr_encoder)
    # comps = (aggr_control, aggr_noise_schedule, aggr_annealing_schedule, aggr_encoder)
    # comps = (aggr_control, aggr_cos_noise_schedule, aggr_annealing_schedule, aggr_encoder)
    # comps = (bca_control, noise_schedule, annealing_schedule, bca_encoder)
    # comps = (bca_control, cos_noise_schedule, annealing_schedule, bca_encoder)
    # comps = (bca_control, bca_noise_schedule, bca_annealing_schedule, bca_encoder)
    # comps = (bca_control, bca_cos_noise_schedule, bca_annealing_schedule, bca_encoder)
    # comps = (mha_control, mha_noise_schedule, mha_annealing_schedule, mha_encoder)

    cdvi = DIS(
        z_dim=config.z_dim,
        num_steps=config.num_steps,
        control=comps[0],
        step_size_schedule=step_size_schedule,
        noise_schedule=comps[1],
        annealing_schedule=comps[2],
        use_score=config.use_score,
        use_error=False,
        device=device,
    )

    # cdvi = CMCD(
    #     z_dim=config.z_dim,
    #     num_steps=config.num_steps,
    #     control=comps[0],
    #     step_size_schedule=step_size_schedule,
    #     noise_schedule=comps[1],
    #     annealing_schedule=comps[2],
    #     device=device,
    # )

    model = DVI(encoder=comps[3], cdvi=cdvi, contextual_target=contextual_target)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    trainer = DVITrainerContext(
        model=model,
        device=device,
        dataset=dataset,
        train_loader=dataloader,
        val_loader=dataloader,
        optimizer=optimizer,
        scheduler=None,
        generator=torch.Generator(device=device),
        wandb_logging=False,
        num_subtasks=32,
        num_samples=32,
        val_grad_off=True,
    )

    trainer.wandb_logging = False
    if trainer.wandb_logging:
        wandb.init(project="local-toytask", name="bca")

    train = False
    if train:
        num_epochs = 3000
        trainer.train(num_epochs=num_epochs, max_clip_norm=None, alpha=None)

    visualize_dvi_2d(device, model, dataset)


if __name__ == "__main__":
    main()
