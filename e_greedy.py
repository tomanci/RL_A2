import numpy as np
from action_slection import ActionSlection

from config import EpsilonConfig

class EpsilonGreedy(ActionSlection):

    def __init__(self, config: EpsilonConfig) -> None:
        self.config = config
        self.epsilon = self.config.initial

    def select_action(self, q_sa):
        if np.random.random() <= self.epsilon:
            return np.argmin(q_sa)
        else:
            return np.argmax(q_sa)
        
    def decay(self) -> None:
        self.epsilon = max(self.config.min, self.epsilon * self.config.decay)
