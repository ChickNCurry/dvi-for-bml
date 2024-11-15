from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class CommonConfig:
    x_dim: int = 1
    y_dim: int = 1
    c_dim: int = 2
    z_dim: int = 2
    h_dim: int = 64
    num_layers: int = 3
    non_linearity: str = "SiLU"


@dataclass
class DVIProcessConfig:
    _target_: str = "src.dvi_process.DIS"
    num_steps: int = 16


@dataclass
class ControlFunctionConfig:
    is_cross_attentive: bool = False


@dataclass
class SetEncoderConfig:
    aggregation: str = "max"
    use_context_size: bool = False
    is_attentive: bool = True


@dataclass
class DecoderConfig:
    has_lat_path: bool = True
    has_det_path: bool = False
    is_cross_attentive: bool = False


@dataclass
class TrainingConfig:
    num_epochs: int = 6000
    batch_size: int = 1024
    learning_rate: float = 3e-4
    wandb_logging: bool = False


@dataclass
class BenchmarkConfig:
    _target_: str = "metalearning_benchmarks.line_sine1d_benchmark.LineSine1D"
    n_task: int = 4096
    n_datapoints_per_task: int = 64
    output_noise: float = 0.1
    seed_task: int = 1237
    seed_x: int = 123
    seed_noise: int = 1237


@dataclass
class Config:
    common: CommonConfig = field(default_factory=CommonConfig)
    dvi_process: DVIProcessConfig = field(default_factory=DVIProcessConfig)
    control_function: ControlFunctionConfig = field(
        default_factory=ControlFunctionConfig
    )
    set_encoder: SetEncoderConfig = field(default_factory=SetEncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
