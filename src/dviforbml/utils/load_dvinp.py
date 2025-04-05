import random
from enum import Enum
from typing import Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from metalearning_benchmarks import MetaLearningBenchmark
from omegaconf import DictConfig
from torch.optim.adamw import AdamW


from torch.utils.data import DataLoader, random_split

from dviforbml.architectures.dvinp import DVINP
from dviforbml.components.cdvi.dis import DIS
from dviforbml.components.cdvi.ula import ULA
from dviforbml.components.cdvi.cmcd import CMCD
from dviforbml.components.control.aggr_control import AggrControl
from dviforbml.components.control.bca_control import BCAControl
from dviforbml.components.control.mhca_control import MHCAControl
from dviforbml.components.decoder.decoder import Decoder
from dviforbml.components.encoder.aggr_encoder import Aggr, AggrEncoder
from dviforbml.components.encoder.bca_encoder import BCAEncoder
from dviforbml.components.encoder.mhca_encoder import MHCAEncoder
from dviforbml.components.schedule.annealing_schedule import (
    AggrAnnealingSchedule,
    AnnealingSchedule,
    BCAAnnealingSchedule,
)
from dviforbml.components.schedule.constr_noise_schedule import (
    AggrConstrNoiseSchedule,
    BCAConstrNoiseSchedule,
    ConstrNoiseSchedule,
    MHCAConstrNoiseSchedule,
)
from dviforbml.components.schedule.free_noise_schedule import (
    AggrFreeNoiseSchedule,
    BCAFreeNoiseSchedule,
    FreeNoiseSchedule,
    MHCAFreeNoiseSchedule,
)
from dviforbml.components.schedule.step_size_schedule import StepSizeSchedule
from dviforbml.training.dvinp_trainer import (
    DVINPTrainer,
    DVINPTrainerContext,
    DVINPTrainerData,
    DVINPTrainerForward,
    DVINPTrainerForwardAndContext,
)
from dviforbml.utils.datasets import MetaLearningDataset, Sinusoid1DFreq
from dviforbml.utils.helper import load_state_dicts


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


class TrainerVariant(Enum):
    DATA = "data"
    CONTEXT = "cntxt"
    FORWARD = "fwd"
    FORWARDANDCONTEXT = "fwdcntxt"


def load_dvinp(
    cfg: DictConfig,
    device: torch.device,
    dir: str | None = None,
    load_decoder_only: bool = False,
    train_decoder: bool = True,
    debugging: bool = False,
) -> Tuple[DVINP, DVINPTrainer, DataLoader, DataLoader]:
    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    g = torch.Generator().manual_seed(cfg.training.seed)

    if (
        cfg.benchmark._target_
        == "metalearning_benchmarks.sinusoid1d_benchmark.Sinusoid1DFreq"
    ):
        benchmark = Sinusoid1DFreq(
            n_task=cfg.benchmark.n_task,
            n_datapoints_per_task=cfg.benchmark.n_datapoints_per_task,
            output_noise=cfg.benchmark.output_noise,
            seed_task=cfg.benchmark.seed_task,
            seed_x=cfg.benchmark.seed_x,
            seed_noise=cfg.benchmark.seed_noise,
        )
    else:
        benchmark: MetaLearningBenchmark = instantiate(cfg.benchmark)

    dataset = MetaLearningDataset(benchmark, cfg.training.max_context_size, g)

    train_set, val_set = (
        random_split(
            dataset,
            [len(dataset) - cfg.training.num_val_tasks, cfg.training.num_val_tasks],
        )
        if not debugging
        else (dataset, dataset)
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=g,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.training.num_val_tasks,
        shuffle=True,
        generator=g,
    )

    test_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=True,
        generator=g,
    )

    context_variant = ContextVariant(cfg.model.context_variant)
    noise_variant = NoiseVariant(cfg.model.noise_variant)
    model_variant = ModelVariant(cfg.model.model_variant)
    trainer_variant = TrainerVariant(cfg.training.trainer_variant)

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
                    )

                case NoiseVariant.CONSTR:
                    noise_schedule = AggrConstrNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        num_steps=cfg.model.num_steps,
                        num_layers=cfg.model.num_layers_sched,
                        non_linearity=cfg.model.non_linearity,
                        max_context_size=cfg.model.max_context_size,
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
                    )

                case NoiseVariant.CONSTR:
                    noise_schedule = BCAConstrNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        num_steps=cfg.model.num_steps,
                        num_layers=cfg.model.num_layers_sched,
                        non_linearity=cfg.model.non_linearity,
                        max_context_size=cfg.model.max_context_size,
                    )

        case ContextVariant.MHCA:
            encoder = MHCAEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_layers=cfg.model.num_layers_enc,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
                num_blocks=cfg.model.num_blocks,
                max_context_size=cfg.model.max_context_size,
            )

            control = MHCAControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers_ctrl,
                non_linearity=cfg.model.non_linearity,
                max_context_size=cfg.model.max_context_size,
                use_score=model_variant == ModelVariant.DIS_SCORE,
                num_heads=cfg.model.cross_attn_num_heads,
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
                    noise_schedule = MHCAFreeNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        num_steps=cfg.model.num_steps,
                        num_layers=cfg.model.num_layers_sched,
                        non_linearity=cfg.model.non_linearity,
                        num_heads=cfg.model.cross_attn_num_heads,
                        max_context_size=cfg.model.max_context_size,
                        device=device,
                    )

                case NoiseVariant.CONSTR:
                    noise_schedule = MHCAConstrNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        num_steps=cfg.model.num_steps,
                        num_layers=cfg.model.num_layers_sched,
                        non_linearity=cfg.model.non_linearity,
                        num_heads=cfg.model.cross_attn_num_heads,
                        max_context_size=cfg.model.max_context_size,
                        device=device,
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
                )

            case NoiseVariant.CONSTR:
                noise_schedule = ConstrNoiseSchedule(
                    z_dim=cfg.model.z_dim,
                    num_steps=cfg.model.num_steps,
                    device=device,
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

    decoder = Decoder(
        x_dim=cfg.model.x_dim,
        z_dim=cfg.model.z_dim,
        h_dim=cfg.model.h_dim_dec,
        y_dim=cfg.model.y_dim,
        num_layers=cfg.model.num_layers_dec,
        non_linearity=cfg.model.non_linearity,
    )

    model = DVINP(
        encoder=encoder,
        cdvi=cdvi,
        decoder=decoder,
    ).to(device)

    params = (
        model.parameters()
        if train_decoder
        else list(model.encoder.parameters()) + list(model.cdvi.parameters())
    )

    optimizer = AdamW(params=params, lr=cfg.training.learning_rate)

    trainer_params = {
        "model": model,
        "device": device,
        "dataset": dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": None,  # scheduler,
        "generator": g,
        "wandb_logging": cfg.wandb.logging,
        "num_subtasks": cfg.training.num_subtasks,
        "num_samples": cfg.training.num_samples,
        "val_grad_off": not model_variant == ModelVariant.DIS_SCORE,
    }

    match trainer_variant:
        case TrainerVariant.CONTEXT:
            trainer = DVINPTrainerContext(**trainer_params)
        case TrainerVariant.DATA:
            trainer = DVINPTrainerData(**trainer_params)
        case TrainerVariant.FORWARD:
            trainer = DVINPTrainerForward(**trainer_params)
        case TrainerVariant.FORWARDANDCONTEXT:
            trainer = DVINPTrainerForwardAndContext(**trainer_params)

    if dir is not None:
        model, trainer = load_state_dicts(dir, model, trainer, load_decoder_only)

    return model, trainer, test_loader, val_loader
