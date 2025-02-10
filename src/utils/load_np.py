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
from torch.utils.data import DataLoader, random_split

from src.components.decoder.decoder import Decoder
from src.components.encoder.aggr_encoder import Aggr, AggrEncoder
from src.components.encoder.bca_encoder import BCAEncoder
from src.components.np import AGGRCNP, AGGRNP, BCACNP, BCANP
from src.train.np_trainer import BetterNPTrainer, NoisyNPTrainer
from src.utils.datasets import MetaLearningDataset


class ContextVariant(Enum):
    MEAN = "mean"
    MAX = "max"
    BCA = "bca"


class ModelVariant(Enum):
    CNP = "cnp"
    NP = "np"


def load_np(
    cfg: DictConfig,
    device: torch.device,
    dir: str | None = None,
) -> Tuple[
    AGGRNP | BCANP | AGGRCNP | BCACNP, NoisyNPTrainer | BetterNPTrainer, DataLoader
]:

    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    g = torch.Generator()
    g.manual_seed(cfg.training.seed)

    benchmark: MetaLearningBenchmark = instantiate(cfg.benchmark)
    dataset = MetaLearningDataset(benchmark, cfg.training.max_context_size)

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
        shuffle=False,
        generator=g,
    )

    test_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        generator=g,
    )

    decoder = Decoder(
        x_dim=cfg.model.x_dim,
        z_dim=cfg.model.z_dim,
        h_dim=cfg.model.h_dim,
        y_dim=cfg.model.y_dim,
        num_layers=cfg.model.num_layers,
        non_linearity=cfg.model.non_linearity,
    )

    context_variant = ContextVariant(cfg.model.context_variant)
    model_variant = ModelVariant(cfg.model.model_variant)

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

            match model_variant:
                case ModelVariant.CNP:

                    model = AGGRCNP(encoder=encoder, decoder=decoder)

                case ModelVariant.NP:

                    model = AGGRNP(encoder=encoder, decoder=decoder)

        case ContextVariant.BCA:

            encoder = BCAEncoder(
                c_dim=cfg.model.c_dim,
                h_dim=cfg.model.h_dim,
                num_layers=cfg.model.num_layers,
                non_linearity=cfg.model.non_linearity,
                num_heads=cfg.model.self_attn_num_heads,
            )

            match model_variant:
                case ModelVariant.CNP:

                    model = BCACNP(encoder=encoder, decoder=decoder)

                case ModelVariant.NP:

                    model = BCANP(encoder=encoder, decoder=decoder)

    model = model.to(device)
    optimizer = AdamW(params=model.parameters(), lr=cfg.training.learning_rate)

    trainer = NoisyNPTrainer(
        model=model,
        device=device,
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        wandb_logging=cfg.wandb.logging,
        num_subtasks=cfg.training.num_subtasks,
        sample_size=cfg.training.sample_size,
    )

    # trainer = BetterNPTrainer(
    #     model=model,
    #     device=device,
    #     dataset=dataset,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     scheduler=None,
    #     wandb_logging=cfg.wandb.logging,
    #     num_subtasks=cfg.training.num_subtasks,
    #     sample_size=cfg.training.sample_size,
    # )

    if dir is not None:

        model_path = f"{dir}/model.pth"
        optim_path = f"{dir}/optim.pth"

        if os.path.exists(model_path):
            model_state_dict = torch.load(
                model_path, map_location=torch.device("cpu"), weights_only=False
            )

            model.load_state_dict(model_state_dict, strict=False)
            print(f"loaded model from {model_path}")

        if os.path.exists(optim_path):
            optim_state_dict = torch.load(
                optim_path, map_location=torch.device("cpu"), weights_only=False
            )

            trainer.optimizer.load_state_dict(optim_state_dict)
            print(f"loaded optim from {optim_path}")

    return model, trainer, test_loader
