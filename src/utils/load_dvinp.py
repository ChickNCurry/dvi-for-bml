import os
import random
from enum import Enum
from typing import Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from metalearning_benchmarks import MetaLearningBenchmark  # type: ignore
from omegaconf import DictConfig
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split

from src.architectures.dvinp import DVINP
from src.components.cdvi.dis import DIS
from src.components.control.aggr_control import AggrControl
from src.components.control.bca_control import BCAControl
from src.components.decoder.decoder import Decoder
from src.components.encoder.aggr_encoder import Aggr, AggrEncoder
from src.components.encoder.bca_encoder import BCAEncoder
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
from src.training.dvinp_trainer import (
    DVINPTrainer,
    DVINPTrainerContext,
    DVINPTrainerData,
    DVINPTrainerForward,
    DVINPTrainerForwardAndContext,
)
from src.utils.datasets import MetaLearningDataset, Sinusoid1DFreq


class ContextVariant(Enum):
    MEAN = "mean"
    MAX = "max"
    BCA = "bca"


class NoiseVariant(Enum):
    FREE = "free"
    COS = "cos"


class ModelVariant(Enum):
    DIS = "dis"
    DIS_SCORE = "dis_score"
    CMCD = "cmcd"


class TrainerVariant(Enum):
    DATA = "data"
    CONTEXT = "context"
    FORWARD = "forward"
    FORWARDANDCONTEXT = "forwardandcontext"


def load_dvinp(
    cfg: DictConfig,
    device: torch.device,
    dir: str | None = None,
    load_decoder_only: bool = False,
    train_decoder: bool = True,
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

    train_set, val_set = random_split(
        dataset, [len(dataset) - cfg.training.num_val_tasks, cfg.training.num_val_tasks]
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
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
                aggregation=Aggr(ContextVariant(context_variant).value),
                max_context_size=cfg.model.max_context_size,
            )

            control = AggrControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                use_score=model_variant == ModelVariant.DIS_SCORE,
            )

            annealing_schedule = AggrAnnealingSchedule(
                h_dim=cfg.model.h_dim,
                non_linearity=cfg.model.non_linearity,
                num_steps=cfg.model.num_steps,
                device=device,
            )

            match noise_variant:
                case NoiseVariant.FREE:

                    noise_schedule = AggrNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        non_linearity=cfg.model.non_linearity,
                        num_steps=cfg.model.num_steps,
                        device=device,
                    )

                case NoiseVariant.COS:

                    noise_schedule = AggrCosineNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        non_linearity=cfg.model.non_linearity,
                        num_steps=cfg.model.num_steps,
                    )

        case ContextVariant.BCA:

            encoder = BCAEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.h_dim,
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
            )

            control = BCAControl(
                h_dim=cfg.model.h_dim,
                z_dim=cfg.model.z_dim,
                num_steps=cfg.model.num_steps,
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                use_score=model_variant == ModelVariant.DIS_SCORE,
            )

            annealing_schedule = BCAAnnealingSchedule(
                h_dim=cfg.model.h_dim,
                non_linearity=cfg.model.non_linearity,
                num_steps=cfg.model.num_steps,
                device=device,
            )

            match noise_variant:
                case NoiseVariant.FREE:

                    noise_schedule = BCANoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        non_linearity=cfg.model.non_linearity,
                        num_steps=cfg.model.num_steps,
                        device=device,
                    )

                case NoiseVariant.COS:

                    noise_schedule = BCACosineNoiseSchedule(
                        z_dim=cfg.model.z_dim,
                        h_dim=cfg.model.h_dim,
                        non_linearity=cfg.model.non_linearity,
                        num_steps=cfg.model.num_steps,
                    )

    if not cfg.model.contextual_schedules:

        annealing_schedule = AnnealingSchedule(
            num_steps=cfg.model.num_steps,
            device=device,
        )

        match noise_variant:
            case NoiseVariant.FREE:

                noise_schedule = NoiseSchedule(
                    z_dim=cfg.model.z_dim,
                    num_steps=cfg.model.num_steps,
                    device=device,
                )

            case NoiseVariant.COS:

                noise_schedule = CosineNoiseSchedule(
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

    decoder = Decoder(
        x_dim=cfg.model.x_dim,
        z_dim=cfg.model.z_dim,
        h_dim=cfg.model.h_dim,
        y_dim=cfg.model.y_dim,
        num_layers=cfg.model.num_layers,
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

    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer=optimizer,
    #     T_0=5,
    #     T_mult=2,
    #     eta_min=cfg.training.learning_rate * 0.01,
    # )

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

        model_path = f"{dir}/model.pth"
        optim_path = f"{dir}/optim.pth"

        if os.path.exists(model_path):
            dvinp_state_dict = torch.load(
                model_path, map_location=torch.device("cpu"), weights_only=False
            )

            if load_decoder_only:
                model.decoder.load_state_dict(
                    {
                        k.split("decoder.")[-1]: v
                        for k, v in dvinp_state_dict.items()
                        if "decoder" in k
                    }
                )
                print(f"loaded decoder from {model_path}")
            else:
                model.load_state_dict(dvinp_state_dict, strict=False)
                print(f"loaded model from {model_path}")
        else:
            print(f"model not found at {model_path}")

        if os.path.exists(optim_path):
            optim_state_dict = torch.load(
                optim_path, map_location=torch.device("cpu"), weights_only=False
            )

            if not load_decoder_only:
                trainer.optimizer.load_state_dict(optim_state_dict)
                print(f"loaded optim from {optim_path}")
        else:
            print(f"optim not found at {optim_path}")

    return model, trainer, test_loader, val_loader
