import numpy as np
import torch


import numpy as np
from action_slection import ActionSlection

from config import TempConfig

class Boltzmann(ActionSlection):

    def __init__(self, config: TempConfig) -> None:
        self.config = config
        self.temp = self.config.initial

    def select_action(self, q_action_values):
        q_action_values = q_action_values / self.temp # scale by temperature
        z = q_action_values - np.max(q_action_values) # substract max to prevent overflow of softmax 
        z_exp = np.exp(z)
        probabilities =  z_exp/np.sum(z_exp) # compute softmax

        assert np.abs(np.sum(probabilities) - 1.0) < 0.001, (
            f"Expected softmax probabilities {probabilities} to sum up to 1 "
            + f"but the actual sum is {np.sum(probabilities)}"
        )
        a = np.random.choice(np.arange(probabilities.shape[0]), p=probabilities)
        return a
        
    def decay(self) -> None:
        self.temp = max(self.config.min, self.temp * self.config.decay)
