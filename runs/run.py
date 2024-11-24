import os

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.components.contextual_dvi import ContextualDVI
from src.components.control import Control
from src.components.decoder import Decoder
from src.components.dvi_process import DiffusionVIProcess
from src.components.encoder import SetEncoder
from src.components.hyper_net import HyperNet
from src.utils.context_datasets import MetaLearningDataset
from src.utils.train import train


@hydra.main(version_base=None, config_name="config", config_path="config")
def run(config: DictConfig) -> None:

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
        is_aggregative=not config.control_and_hyper_net.is_cross_attentive
        or not config.decoder.is_cross_attentive,
        is_non_aggregative=config.control_and_hyper_net.is_cross_attentive
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
        is_cross_attentive=config.control_and_hyper_net.is_cross_attentive,
        num_heads=config.control_and_hyper_net.num_heads,
    )

    hyper_net = (
        HyperNet(
            h_dim=config.common.h_dim,
            z_dim=config.common.z_dim,
            non_linearity=config.common.non_linearity,
            num_steps=config.dvi_process.num_steps,
            is_cross_attentive=config.control_and_hyper_net.is_cross_attentive,
            num_heads=config.control_and_hyper_net.num_heads,
        )
        if config.control_and_hyper_net.use_hyper_net
        else None
    )

    dvi_process: DiffusionVIProcess = instantiate(
        config.dvi_process,
        z_dim=config.common.z_dim,
        control=control,
        hyper_net=hyper_net,
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

    optimizer = torch.optim.AdamW(  # type: ignore
        contextual_dvi.parameters(), lr=config.training.learning_rate
    )

    if config.wandb.logging:
        wandb.init(project=config.wandb.project, config=OmegaConf.to_container(config))  # type: ignore

    train(
        device=device,
        contextual_dvi=contextual_dvi,
        target_constructor=None,
        num_epochs=config.training.num_epochs,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=None,
        max_clip_norm=config.training.max_clip_norm,
        wandb_logging=config.wandb.logging,
        alpha=config.training.alpha,
    )

    if wandb.run is not None:

        if not os.path.exists("models"):
            os.mkdir("models")

        dir = os.path.join("models", wandb.run.name)
        os.mkdir(dir)

        with open(os.path.join(dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)

        model_path = os.path.join(dir, "cdvi.pth")
        optimizer_path = os.path.join(dir, "optim.pth")

        torch.save(contextual_dvi.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)

        if config.wandb.logging and wandb.run is not None:
            wandb.run.log_model(path=model_path, name=f"{wandb.run.name}_cdvi.pth")
            wandb.run.log_model(path=optimizer_path, name=f"{wandb.run.name}_optim.pth")


if __name__ == "__main__":
    run()
