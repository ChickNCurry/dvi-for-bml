from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class CommonConfig:
    x_dim: int = 1
    y_dim: int = 1
    c_dim: int = 2
    z_dim: int = 2
    h_dim: int = 128
    num_layers: int = 6
    non_linearity: str = "SiLU"


@dataclass
class DVIProcessConfig:
    _target_: str = "src.components.dvi_process.DIS"
    num_steps: int = 16
    has_hyper_net: bool = False


@dataclass
class ControlConfig:
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
    num_epochs: int = 5000
    batch_size: int = 1024
    learning_rate: float = 3e-4
    max_clip_norm: float = 0.2
    plateau_factor: float = 0.3
    plateau_patience: int = 200
    wandb_logging: bool = True


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
class CpuConfig:
    partition: str = "single"
    timeout_min: int = 300
    mem_per_cpu: int = 4000


@dataclass
class GpuConfig:
    partition: str = "gpu_4"
    gres: str = "gpu:1"
    timeout_min: int = 300
    mem_per_gpu: int = 4000


@dataclass
class GpuDevConfig:
    partition: str = "dev_gpu_4"
    gres: str = "gpu:1"
    timeout_min: int = 30
    mem_per_gpu: int = 4000


@dataclass
class HydraConfig:
    mode: str = "MULTIRUN"
    launcher: GpuDevConfig = field(default_factory=GpuDevConfig)


@dataclass
class Config:
    common: CommonConfig = field(default_factory=CommonConfig)
    dvi_process: DVIProcessConfig = field(default_factory=DVIProcessConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    set_encoder: SetEncoderConfig = field(default_factory=SetEncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    hydra: HydraConfig = field(default_factory=HydraConfig)


cs = ConfigStore.instance()
# cs.store(name="dvi_process", node=DVIProcessConfig)
# cs.store(name="set_encoder", node=SetEncoderConfig)
# cs.store(name="control", node=ControlConfig)
# cs.store(name="decoder", node=DecoderConfig)
# cs.store(name="training", node=TrainingConfig)
# cs.store(name="benchmark", node=BenchmarkConfig)
# cs.store(name="hydra", node=HydraConfig)
# cs.store(name="gpu_dev", node=GpuDevConfig)
# cs.store(name="gpu", node=GpuConfig)
# cs.store(name="cpu", node=CpuConfig)
cs.store(name="config", node=Config)
