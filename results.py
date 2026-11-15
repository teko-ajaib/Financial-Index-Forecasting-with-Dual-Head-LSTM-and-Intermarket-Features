from dataclasses import dataclass, field
from typing import Dict, List, Any
import torch


@dataclass
class TrainResult:
    model: torch.nn.Module
    scaler: Any
    history: Dict[str, list]
    feature_names: List[str]
    num_features: int
    target_ticker: str
    target_idx: int
    hyperparameters: dict
    best_val_loss: float
    best_val_acc: float
    best_epoch: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
