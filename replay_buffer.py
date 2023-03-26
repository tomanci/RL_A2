from collections import deque
import random
from typing import List

from transition import Transition


class ReplayBuffer:

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = deque([], maxlen=buffer_size)

    def add_transition(self, transition: Transition):
        self.buffer.appendleft(transition)

    def get_n_samples(self, n) -> List[Transition]:
        return random.sample(self.buffer, n)

    def size(self) -> int:
        return len(self.buffer)
