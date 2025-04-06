from dataclasses import asdict, dataclass
import json
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

import wandb
from dviforbml.architectures.dvi import DVI
from dviforbml.components.cdvi.cmcd import CMCD
from dviforbml.components.cdvi.dis import DIS
from dviforbml.components.control.aggr_control import AggrControl
from dviforbml.components.control.abstract_control import AbstractControl
from dviforbml.components.control.bca_control import BCAControl
from dviforbml.components.control.mhca_control import MHCAControl
from dviforbml.components.encoder.aggr_encoder import Aggr, AggrEncoder
from dviforbml.components.encoder.abstract_encoder import AbstractEncoder
from dviforbml.components.encoder.bca_encoder import BCAEncoder
from dviforbml.components.encoder.mhca_encoder import MHCAEncoder
from dviforbml.components.schedule.annealing_schedule import (
    AggrAnnealingSchedule,
    AnnealingSchedule,
    BCAAnnealingSchedule,
)
from dviforbml.components.schedule.abstract_schedule import AbstractSchedule
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
from dviforbml.evaluation.visualization.visualize_dvi_contour import (
    visualize_dvi_2d_contour_all,
)
from dviforbml.training.dvi_trainer import DVITrainerContext
from dviforbml.utils.datasets import ContextSetDataset
from dviforbml.utils.distros import TaskPosteriorGMM
from dviforbml.utils.hash import get_object_hash


def save(obj: object) -> None:
    torch.save(
        obj.state_dict(),
        f"scripts/training/dvi/state_dicts/{get_object_hash(obj)}.pth",
    )


def load(obj: object) -> None:
    obj.load_state_dict(
        torch.load(
            f"scripts/training/dvi/state_dicts/{get_object_hash(obj)}.pth",
            weights_only=False,
        )
    )
    print(f"{obj.__class__.__name__} loaded")


def run() -> None:
    device = torch.device("cpu")

    @dataclass
    class Config:
        num_steps: int = 16
        c_dim: int = 2
        z_dim: int = 2
        h_dim: int = 32
        num_layers: int = 4
        non_linearity: str = "GELU"
        num_heads_sa: int = None  # 8
        num_heads_ca: int = 4
        num_blocks: int = 1
        aggr: str = "mean"
        use_score: bool = False
        learning_rate: float = 1e-3
        size: int = 64
        batch_size: int = 64
        max_context_size: int = None  # 10  # None
        num_epochs: int = 5000
        max_clip_norm: float = 1.0

    cfg = Config()

    dataset = ContextSetDataset(
        size=cfg.size,
        c_dim=cfg.c_dim,
        max_context_size=10,
        sampling_factor=4,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True)

    contextual_target = TaskPosteriorGMM

    # Encoders

    aggr_encoder = AggrEncoder(
        c_dim=cfg.c_dim,
        h_dim=cfg.h_dim,
        z_dim=cfg.z_dim,
        num_layers=cfg.num_layers,
        non_linearity=cfg.non_linearity,
        num_heads=cfg.num_heads_sa,
        num_blocks=cfg.num_blocks,
        max_context_size=cfg.max_context_size,
        aggregation=Aggr(cfg.aggr),
    )

    bca_encoder = BCAEncoder(
        c_dim=cfg.c_dim,
        h_dim=cfg.h_dim,
        z_dim=cfg.z_dim,
        num_layers=cfg.num_layers,
        non_linearity=cfg.non_linearity,
        num_heads=cfg.num_heads_sa,
        num_blocks=cfg.num_blocks,
        max_context_size=cfg.max_context_size,
        bca_dim=cfg.h_dim,
    )

    mhca_encoder = MHCAEncoder(
        z_dim=cfg.z_dim,
        c_dim=cfg.c_dim,
        h_dim=cfg.h_dim,
        num_layers=cfg.num_layers,
        non_linearity=cfg.non_linearity,
        num_heads=cfg.num_heads_sa,
        num_blocks=cfg.num_blocks,
        max_context_size=cfg.max_context_size,
    )

    # Controls

    aggr_control = AggrControl(
        h_dim=cfg.h_dim,
        z_dim=cfg.z_dim,
        num_layers=cfg.num_layers,
        non_linearity=cfg.non_linearity,
        max_context_size=cfg.max_context_size,
        num_steps=cfg.num_steps,
        use_score=cfg.use_score,
    )

    bca_control = BCAControl(
        h_dim=cfg.h_dim,
        z_dim=cfg.z_dim,
        num_layers=cfg.num_layers,
        non_linearity=cfg.non_linearity,
        max_context_size=cfg.max_context_size,
        num_steps=cfg.num_steps,
        use_score=cfg.use_score,
    )

    mhca_control = MHCAControl(
        h_dim=cfg.h_dim,
        z_dim=cfg.z_dim,
        num_layers=cfg.num_layers,
        non_linearity=cfg.non_linearity,
        max_context_size=cfg.max_context_size,
        num_steps=cfg.num_steps,
        use_score=cfg.use_score,
        num_heads=cfg.num_heads_ca,
    )

    # Free Noise Schedules

    free_noise_sched = FreeNoiseSchedule(
        z_dim=cfg.z_dim,
        num_steps=cfg.num_steps,
        device=device,
    )

    aggr_free_noise_sched = AggrFreeNoiseSchedule(
        z_dim=cfg.z_dim,
        h_dim=cfg.h_dim,
        non_linearity=cfg.non_linearity,
        num_layers=cfg.num_layers,
        num_steps=cfg.num_steps,
        max_context_size=cfg.max_context_size,
        device=device,
    )

    bca_free_noise_sched = BCAFreeNoiseSchedule(
        z_dim=cfg.z_dim,
        h_dim=cfg.h_dim,
        non_linearity=cfg.non_linearity,
        num_layers=cfg.num_layers,
        num_steps=cfg.num_steps,
        max_context_size=cfg.max_context_size,
        device=device,
    )

    mhca_free_noise_sched = MHCAFreeNoiseSchedule(
        z_dim=cfg.z_dim,
        h_dim=cfg.h_dim,
        non_linearity=cfg.non_linearity,
        num_layers=cfg.num_layers,
        num_steps=cfg.num_steps,
        num_heads=cfg.num_heads_ca,
        max_context_size=cfg.max_context_size,
        device=device,
    )

    # Constr Noise Schedules

    constr_noise_sched = ConstrNoiseSchedule(
        z_dim=cfg.z_dim,
        num_steps=cfg.num_steps,
        device=device,
    )

    aggr_constr_noise_sched = AggrConstrNoiseSchedule(
        z_dim=cfg.z_dim,
        h_dim=cfg.h_dim,
        non_linearity=cfg.non_linearity,
        num_layers=cfg.num_layers,
        num_steps=cfg.num_steps,
        max_context_size=cfg.max_context_size,
    )

    bca_constr_noise_sched = BCAConstrNoiseSchedule(
        z_dim=cfg.z_dim,
        h_dim=cfg.h_dim,
        non_linearity=cfg.non_linearity,
        num_layers=cfg.num_layers,
        num_steps=cfg.num_steps,
        max_context_size=cfg.max_context_size,
    )

    mhca_constr_noise_sched = MHCAConstrNoiseSchedule(
        z_dim=cfg.z_dim,
        h_dim=cfg.h_dim,
        non_linearity=cfg.non_linearity,
        num_layers=cfg.num_layers,
        num_steps=cfg.num_steps,
        num_heads=cfg.num_heads_ca,
        max_context_size=cfg.max_context_size,
        device=device,
    )

    # Annealing Schedules

    # anneal_sched = AnnealingSchedule(num_steps=cfg.num_steps, device=device)

    # aggr_anneal_schedule = AggrAnnealingSchedule(
    #     z_dim=cfg.z_dim,
    #     h_dim=cfg.h_dim,
    #     num_steps=cfg.num_steps,
    #     num_layers=cfg.num_layers,
    #     non_linearity=cfg.non_linearity,
    #     max_context_size=cfg.max_context_size,
    #     device=device,
    # )

    # bca_anneal_schedule = BCAAnnealingSchedule(
    #     z_dim=cfg.z_dim,
    #     h_dim=cfg.h_dim,
    #     num_steps=cfg.num_steps,
    #     num_layers=cfg.num_layers,
    #     non_linearity=cfg.non_linearity,
    #     max_context_size=cfg.max_context_size,
    #     device=device,
    # )

    # Step Size Schedules

    step_size_schedule = StepSizeSchedule(num_steps=cfg.num_steps, device=device)

    # SAVE

    save(aggr_encoder)
    save(bca_encoder)
    save(mhca_encoder)

    save(aggr_control)
    save(bca_control)
    save(mhca_control)

    save(free_noise_sched)
    save(aggr_free_noise_sched)
    save(bca_free_noise_sched)
    save(mhca_free_noise_sched)

    save(constr_noise_sched)
    save(aggr_constr_noise_sched)
    save(bca_constr_noise_sched)
    save(mhca_constr_noise_sched)

    # Architectures

    arches: List[
        Tuple[AbstractControl, AbstractSchedule, AbstractSchedule, AbstractEncoder]
    ] = [
        (bca_encoder, bca_control, free_noise_sched, None),
        (bca_encoder, bca_control, constr_noise_sched, None),
        (bca_encoder, bca_control, bca_free_noise_sched, None),
        (bca_encoder, bca_control, bca_constr_noise_sched, None),
        (aggr_encoder, aggr_control, free_noise_sched, None),
        (aggr_encoder, aggr_control, constr_noise_sched, None),
        (aggr_encoder, aggr_control, aggr_free_noise_sched, None),
        (aggr_encoder, aggr_control, aggr_constr_noise_sched, None),
        (mhca_encoder, mhca_control, free_noise_sched, None),
        (mhca_encoder, mhca_control, constr_noise_sched, None),
        (mhca_encoder, mhca_control, mhca_free_noise_sched, None),
        (mhca_encoder, mhca_control, mhca_constr_noise_sched, None),
    ]

    for comps in arches:
        for c in comps:
            if c is not None:
                load(c)

        id = "-".join([f"{c.__class__.__name__}" for c in comps])

        cdvi = DIS(
            z_dim=cfg.z_dim,
            num_steps=cfg.num_steps,
            control=comps[1],
            step_size_schedule=step_size_schedule,
            noise_schedule=comps[2],
            annealing_schedule=comps[3],
            use_score=cfg.use_score,
            device=device,
        )

        # cdvi = CMCD(
        #     z_dim=cfg.z_dim,
        #     num_steps=cfg.num_steps,
        #     control=comps[1],
        #     step_size_schedule=step_size_schedule,
        #     noise_schedule=comps[2],
        #     annealing_schedule=comps[3],
        #     device=device,
        # )

        # cdvi = ULA(
        #     z_dim=cfg.z_dim,
        #     num_steps=cfg.num_steps,
        #     step_size_schedule=step_size_schedule,
        #     noise_schedule=comps[2],
        #     annealing_schedule=comps[3],
        #     device=device,
        # )

        model = DVI(
            encoder=comps[0], cdvi=cdvi, contextual_target=contextual_target
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

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

        trainer.wandb_logging = True
        if trainer.wandb_logging:
            wandb.init(
                project="local-toytask-auto",
                name=id,
                config=asdict(cfg),
            )

        trainer.train(num_epochs=cfg.num_epochs, max_clip_norm=None, alpha=None)

        visualize_dvi_2d_contour_all(
            device,
            model,
            dataset,
            save_fig_path=f"scripts/training/dvi/pdf/{id}.pdf",
            save_csv_path=f"scripts/training/dvi/csv/{id}.csv",
            id=id,
        )

        with open(f"scripts/training/dvi/configs/{id}.json", "w") as f:
            json.dump(asdict(cfg), f, indent=4)

        wandb.finish()


if __name__ == "__main__":
    run()
