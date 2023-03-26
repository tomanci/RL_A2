from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: int
    next_state: np.ndarray
    terminated: bool
