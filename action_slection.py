from abc import ABC, abstractmethod
import numpy as np

class ActionSlection(ABC):
    
    @abstractmethod
    def select_action(self, q_sa: np.ndarray) -> int:
        pass

    @abstractmethod
    def decay(self) -> None:
        pass