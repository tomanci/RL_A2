import numpy as np
from action_slection import ActionSlection


class RandomPolicy(ActionSlection):


    def select_action(self, q_sa):
        return np.random.randint(0, q_sa.shape[0])
        
    def decay(self) -> None:
        pass
