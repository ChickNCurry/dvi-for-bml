import os

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from runs.config.config import Config
from src.context_datasets import MetaLearningDataset
from src.contextual_dvi import ContextualDVI
from src.control import Control
from src.decoder import Decoder
from src.dvi_process import DiffusionVIProcess
from src.encoder import SetEncoder
from src.train import train


@hydra.main(version_base=None, config_name="config")
def run(config: Config) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmark = instantiate(config.benchmark)
    dataset = MetaLearningDataset(benchmark=benchmark)
    dataloader = DataLoader(dataset, config.training.batch_size, True)

    set_encoder = SetEncoder(
        c_dim=config.common.c_dim,
        h_dim=config.common.h_dim,
        num_layers=config.common.num_layers,
        non_linearity=config.common.non_linearity,
        is_attentive=config.set_encoder.is_attentive,
        is_aggregative=not config.control.is_cross_attentive
        or not config.decoder.is_cross_attentive,
        is_non_aggregative=config.control.is_cross_attentive
        or config.decoder.is_cross_attentive,
        use_context_size=config.set_encoder.use_context_size,
        aggregation=config.set_encoder.aggregation,
        max_context_size=dataset.max_context_size,
    )

    control = Control(
        h_dim=config.common.h_dim,
        z_dim=config.common.z_dim,
        num_layers=config.common.num_layers,
        non_linearity=config.common.non_linearity,
        num_steps=config.dvi_process.num_steps,
        is_cross_attentive=config.control.is_cross_attentive,
    )

    dvi_process: DiffusionVIProcess = instantiate(
        config.dvi_process,
        z_dim=config.common.z_dim,
        control=control,
        device=device,
    )

    decoder = Decoder(
        x_dim=config.common.x_dim,
        z_dim=config.common.z_dim,
        h_dim=config.common.h_dim,
        y_dim=config.common.y_dim,
        num_layers=config.common.num_layers,
        non_linearity=config.common.non_linearity,
        has_lat_path=config.decoder.has_lat_path,
        has_det_path=config.decoder.has_det_path,
        is_cross_attentive=config.decoder.is_cross_attentive,
    )

    contextual_dvi = ContextualDVI(
        encoder=set_encoder,
        dvi_process=dvi_process,
        decoder=decoder,
    ).to(device)

    optimizer = torch.optim.Adam(  # type: ignore
        contextual_dvi.parameters(), lr=config.training.learning_rate
    )

    if config.training.wandb_logging:
        wandb.init(project="dvi-for-bml", config=OmegaConf.to_container(config))  # type: ignore

    train(
        device=device,
        contextual_dvi=contextual_dvi,
        target_constructor=None,
        num_epochs=config.training.num_epochs,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=None,
        wandb_logging=config.training.wandb_logging,
    )

    if config.training.wandb_logging and wandb.run is not None:

        dir = os.path.join("models", wandb.run.name)
        os.mkdir(dir)

        with open(os.path.join(dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)

        path = os.path.join(dir, f"{contextual_dvi}.pth")

        torch.save(contextual_dvi.state_dict(), path)

        artifact = wandb.Artifact(f"{wandb.run.name}.pth", type=wandb.run.name)
        artifact.add_file(path)

        wandb.run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    run()
