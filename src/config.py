from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class Config:

    def __init__(
        self,
        num_steps: int,
        x_dim: int,
        y_dim: int,
        c_dim: int,
        z_dim: int,
        h_dim: int,
        num_layers: int,
        non_linearity: str,
        learning_rate: float,
        batch_size: int,
        is_attentive: bool,
        is_control_cross_attentive: bool,
        is_decoder_cross_attentive: bool,
        aggregation: str,
        use_context_size: bool,
        has_lat_path: bool,
        has_det_path: bool,
    ) -> None:
        self.num_steps = num_steps
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.is_attentive = is_attentive
        self.is_control_cross_attentive = is_control_cross_attentive
        self.is_decoder_cross_attentive = is_decoder_cross_attentive
        self.aggregation = aggregation
        self.use_context_size = use_context_size
        self.has_lat_path = has_lat_path
        self.has_det_path = has_det_path

    def to_dict(self) -> Dict[Any, Any]:
        return asdict(self)
