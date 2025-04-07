import random
from enum import Enum
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim.adamw import AdamW


from torch.utils.data import DataLoader

from dviforbml.architectures.dvi import DVI
from dviforbml.components.cdvi.dis import DIS
from dviforbml.components.cdvi.ula import ULA
from dviforbml.components.cdvi.cmcd import CMCD
from dviforbml.components.control.aggr_control import AggrControl
from dviforbml.components.control.bca_control import BCAControl
from dviforbml.components.encoder.aggr_encoder import Aggr, AggrEncoder
from dviforbml.components.encoder.bca_encoder import BCAEncoder
from dviforbml.components.schedule.annealing_schedule import (
    AggrAnnealingSchedule,
    AnnealingSchedule,
    BCAAnnealingSchedule,
)
from dviforbml.components.schedule.constr_noise_schedule import (
    AggrConstrNoiseSchedule,
    BCAConstrNoiseSchedule,
    ConstrNoiseSchedule,
)
from dviforbml.components.schedule.free_noise_schedule import (
    AggrFreeNoiseSchedule,
    BCAFreeNoiseSchedule,
    FreeNoiseSchedule,
)
from dviforbml.components.schedule.step_size_schedule import StepSizeSchedule
from dviforbml.training.dvi_trainer import DVITrainer, DVITrainerContext
from dviforbml.utils.datasets import ContextSetDataset
from dviforbml.utils.distros import TaskPosteriorGMM


class ContextVariant(Enum):
    MEAN = "mean"
    MAX = "max"
    BCA = "bca"
    MHCA = "mhca"


class NoiseVariant(Enum):
    FREE = "free"
    CONSTR = "constr"


class ModelVariant(Enum):
    DIS = "dis"
    DIS_SCORE = "dis_score"
    CMCD = "cmcd"
    ULA = "ula"


def load_dvi(
    cfg: DictConfig, device: torch.device
) -> Tuple[DVI, DVITrainer, DataLoader]:
    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    g = torch.Generator().manual_seed(cfg.training.seed)

    dataset = ContextSetDataset(
        size=cfg.training.size,
        c_dim=cfg.model.c_dim,
        max_context_size=cfg.training.max_context_size,
        sampling_factor=4,
        generator=g,
    )

    dataloader = DataLoader(
        dataset=dataset, batch_size=cfg.training.batch_size, shuffle=True, generator=g
    )

    context_variant = ContextVariant(cfg.model.context_variant)
    noise_variant = NoiseVariant(cfg.model.noise_variant)
    model_variant = ModelVariant(cfg.model.model_variant)

    contextual_target = TaskPosteriorGMM

    match context_variant:
        case ContextVariant.MEAN | ContextVariant.MAX:
            encoder = AggrEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_layers=cfg.model.num_layers_enc,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
                num_blocks=cfg.model.num_blocks,
                max_context_size=cfg.model.max_context_size,
                aggregation=Aggr(ContextVariant(context_variant).value),
            )

            control = AggrControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers_ctrl,
                non_linearity=cfg.model.non_linearity,
                max_context_size=cfg.model.max_context_size,
                use_score=model_variant == ModelVariant.DIS_SCORE,
            )

            annealing_schedule = (
                AggrAnnealingSchedule(
                    z_dim=cfg.model.z_dim,
                    h_dim=cfg.model.h_dim,
                    num_steps=cfg.model.num_steps,
                    num_layers=cfg.model.num_layers_sched,
                    non_linearity=cfg.model.non_linearity,
                    max_context_size=cfg.model.max_context_size,
                    device=device,
                )
                if model_variant is not ModelVariant.DIS
                else None
            )

            match noise_variant:
                case NoiseVariant.FREE:
                    noise_schedule = AggrFreeNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        num_steps=cfg.model.num_steps,
                        num_layers=cfg.model.num_layers_sched,
                        non_linearity=cfg.model.non_linearity,
                        max_context_size=cfg.model.max_context_size,
                        device=device,
                        min=cfg.model.noise_min,
                        max=cfg.model.noise_max,
                    )

                case NoiseVariant.CONSTR:
                    noise_schedule = AggrConstrNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        num_steps=cfg.model.num_steps,
                        num_layers=cfg.model.num_layers_sched,
                        non_linearity=cfg.model.non_linearity,
                        max_context_size=cfg.model.max_context_size,
                        min=cfg.model.noise_min,
                        max=cfg.model.noise_max,
                    )

        case ContextVariant.BCA:
            encoder = BCAEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_layers=cfg.model.num_layers_enc,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
                num_blocks=cfg.model.num_blocks,
                max_context_size=cfg.model.max_context_size,
                bca_dim=cfg.model.h_dim,
            )

            control = BCAControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers_ctrl,
                non_linearity=cfg.model.non_linearity,
                max_context_size=cfg.model.max_context_size,
                use_score=model_variant == ModelVariant.DIS_SCORE,
            )

            annealing_schedule = (
                BCAAnnealingSchedule(
                    z_dim=cfg.model.z_dim,
                    h_dim=cfg.model.h_dim,
                    num_steps=cfg.model.num_steps,
                    num_layers=cfg.model.num_layers_sched,
                    non_linearity=cfg.model.non_linearity,
                    max_context_size=cfg.model.max_context_size,
                    device=device,
                )
                if model_variant is not ModelVariant.DIS
                else None
            )

            match noise_variant:
                case NoiseVariant.FREE:
                    noise_schedule = BCAFreeNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        num_steps=cfg.model.num_steps,
                        num_layers=cfg.model.num_layers_sched,
                        non_linearity=cfg.model.non_linearity,
                        max_context_size=cfg.model.max_context_size,
                        device=device,
                        min=cfg.model.noise_min,
                        max=cfg.model.noise_max,
                    )

                case NoiseVariant.CONSTR:
                    noise_schedule = BCAConstrNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        num_steps=cfg.model.num_steps,
                        num_layers=cfg.model.num_layers_sched,
                        non_linearity=cfg.model.non_linearity,
                        max_context_size=cfg.model.max_context_size,
                        min=cfg.model.noise_min,
                        max=cfg.model.noise_max,
                    )

    if not cfg.model.contextual_schedules:
        annealing_schedule = (
            AnnealingSchedule(
                num_steps=cfg.model.num_steps,
                device=device,
            )
            if model_variant is not ModelVariant.DIS
            else None
        )

        match noise_variant:
            case NoiseVariant.FREE:
                noise_schedule = FreeNoiseSchedule(
                    z_dim=cfg.model.z_dim,
                    num_steps=cfg.model.num_steps,
                    device=device,
                    min=cfg.model.noise_min,
                    max=cfg.model.noise_max,
                )

            case NoiseVariant.CONSTR:
                noise_schedule = ConstrNoiseSchedule(
                    z_dim=cfg.model.z_dim,
                    num_steps=cfg.model.num_steps,
                    device=device,
                    min=cfg.model.noise_min,
                    max=cfg.model.noise_max,
                )

    step_size_schedule = StepSizeSchedule(
        num_steps=cfg.model.num_steps,
        device=device,
    )

    match model_variant:
        case ModelVariant.DIS | ModelVariant.DIS_SCORE:
            cdvi = DIS(
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                control=control,
                step_size_schedule=step_size_schedule,
                noise_schedule=noise_schedule,
                annealing_schedule=annealing_schedule,
                use_score=model_variant == ModelVariant.DIS_SCORE,
                device=device,
            )

        case ModelVariant.CMCD:
            cdvi = CMCD(
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                control=control,
                step_size_schedule=step_size_schedule,
                noise_schedule=noise_schedule,
                annealing_schedule=annealing_schedule,
                device=device,
            )

        case ModelVariant.ULA:
            cdvi = ULA(
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                step_size_schedule=step_size_schedule,
                noise_schedule=noise_schedule,
                annealing_schedule=annealing_schedule,
                device=device,
            )

    model = DVI(
        encoder=encoder,
        cdvi=cdvi,
        contextual_target=contextual_target,
    ).to(device)

    optimizer = AdamW(params=model.parameters(), lr=cfg.training.learning_rate)

    trainer = DVITrainerContext(
        model=model,
        device=device,
        dataset=dataset,
        train_loader=dataloader,
        val_loader=dataloader,
        optimizer=optimizer,
        scheduler=None,
        generator=g,
        wandb_logging=cfg.wandb.logging,
        num_subtasks=cfg.training.num_subtasks,
        num_samples=cfg.training.num_samples,
        val_grad_off=not model_variant == ModelVariant.DIS_SCORE,
    )

    return model, trainer, dataloader
