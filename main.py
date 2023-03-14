import gymnasium as gym
import numpy as np
from e-greedy import epsilongreedy
environment = gym.make("CartPole-v1")

MAX_EPOCHS = 1000

def train_qlearn(env, q_net, alpha=0.001, gamma=1.0, epsilon=0.05):
    s, info = env.reset() # s = s_0

    for epoch in range(MAX_EPOCHS):
        sum_sq = 0
        terminated = False

        while not terminated:
            q_sa = [q_net.forward_pass(s, action) for action in env.action_space]
            a = epsilongreedy(q_sa, epsilon) # Select an action
            sp, reward, terminated, truncated, info = env.step(a) # Take the step
            output = q_sa[a]
            q_spa = [q_net.forward_pass(sp, action) for action in env.action_space]
            target = reward + gamma * np.max(q_spa)
            sum_sq += (target - output) ** 2
            s = sp

        # Update the network
        grad = q_net.gradient(sum_sq)
        q_net.backward_pass(grad, alpha)

    return q_net
