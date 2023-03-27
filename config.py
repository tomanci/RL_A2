from dataclasses import dataclass

import numpy as np
from numpy import ndarray
import dataconf

@dataclass
class ArchitectureConfig:
    general_style: str = "Fully connected NN"
    hidden_layers: int = 2
    hidden_layer_initial_size: int = 64
    hidden_layer_scaling: float = 2.0
    activation_function: str = "ReLu"

@dataclass
class EpsilonConfig:
    initial: float = 0.2
    decay: float = 0.99
    min: float = 0.01

@dataclass
class TempConfig:
    initial: float = 0.1
    decay: float = 0.99
    min: float = 0.05

@dataclass
class Config:
    architecture: ArchitectureConfig = ArchitectureConfig()
    nn_architecture: list[int] = list[1, 1, 1, 1]
    learning_rate: float = 0.001
    epochs: int = 1000
    sampling_rate: int = 2
    buffer_size: int = 100000
    policy: str = "epsilon_greedy"
    epsilon: EpsilonConfig = EpsilonConfig()
    temp: TempConfig = TempConfig()
    gamma: float = 0.9


def load_from_yaml(file_path: str):
    config = dataconf.file(file_path, Config)
    return config
