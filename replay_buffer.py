from collections import deque
import random


class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque([], maxlen=buffer_size)

    def add_transition(self, transition):
        self.buffer.appendleft(transition)

    def get_n_samples(self, n):
        return random.sample(self.buffer, n)

    def size(self) -> int:
        return len(self.buffer)
