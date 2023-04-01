from dataclasses import dataclass, field

import dataconf

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
    nn_architecture: list[int] = field(default_factory=lambda: [4,32,32,32,32,2])
    learning_rate: float = 0.001
    learning_rate_decay: float = 0.99 # 1.0 is no decay
    epochs: int = 1000
    sampling_rate: int = 32
    buffer_size: int = 10000
    policy: str = "epsilon_greedy"
    epsilon: EpsilonConfig = EpsilonConfig()
    temp: TempConfig = TempConfig()
    gamma: float = 0.9
    target_network_sync_freq: int = 2  # interval of copying TargetNet to PolicyNet. Only used if TN is actually active.


def load_from_yaml(file_path: str) -> Config:
    print("Loading config from", file_path)
    config = dataconf.file(file_path, Config)
    return config
