import gymnasium as gym
import numpy as np
import torch
from model import DQN
from e_greedy import epsilon_greedy

environment = gym.make("CartPole-v1")

MAX_EPOCHS = 10000
EPSILON_DECAY = 0.99

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def record_episode(env, q_net, epsilon, gamma=0.99):
    s, info = env.reset()
    terminated = False
    states = [s]
    targets = []

    while not terminated:
        q_sa = q_net.forward_pass_no_grad(s).numpy()
        a = epsilon_greedy(q_sa, epsilon)  # Select an action
        sp, reward, terminated, truncated, info = env.step(a)  # Take the step

        q_spa = q_net.forward_pass_no_grad(sp)
        target = reward + gamma * torch.max(q_spa).item()
        q_sa[a] = target

        states.append(sp)
        targets.append(q_sa)

        s = sp

        if truncated:
            break

    return states, targets


def train_model(model_input, targets):
    q_net.optimizer.zero_grad()
    batch_output = q_net.forward_pass(model_input)

    loss = q_net.compute_loss(batch_output, targets)
    loss.backward()
    q_net.optimizer.step()

    return loss.item()


def train_qlearn(env, q_net, alpha=0.001, gamma=1.0, epsilon=0.3):
    max_length = 0
    for epoch in range(MAX_EPOCHS):
        states, targets = record_episode(env, q_net, epsilon, gamma)
        model_input = torch.tensor(states[:-1])
        targets = torch.tensor(targets)
        loss = train_model(model_input, targets)
        episode_length = len(states) - 1

        max_length = max(max_length, episode_length)

        epsilon = max(epsilon * EPSILON_DECAY, 0.01)

        if epoch % 100 == 99:
            print(f"Max length: {loss}")

    print(f"{max_length}")

    # Train a network

    return q_net


q_net = DQN()

train_qlearn(environment, q_net)
