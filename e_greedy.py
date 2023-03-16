import numpy as np


def epsilon_greedy(q_sa, epsilon):
    if np.random.random() <= epsilon:
        return np.argmin(q_sa)
    else:
        return np.argmax(q_sa)