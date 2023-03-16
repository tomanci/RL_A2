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
    actions = []
    rewards = []
    targets = []

    while not terminated:
        q_sa = [q_net.forward_pass_no_grad(s, action).item() for action in range(env.action_space.n)]
        a = epsilon_greedy(q_sa, epsilon)  # Select an action
        sp, reward, terminated, truncated, info = env.step(a)  # Take the step

        q_spa = [q_net.forward_pass_no_grad(sp, action).item() for action in range(env.action_space.n)]
        target = reward + gamma * max(q_spa)

        states.append(sp)
        actions.append(a)
        rewards.append(reward)
        targets.append([target])

        s = sp

        if truncated:
            break

    return states, actions, rewards, targets


def train_qlearn(env, q_net, alpha=0.001, gamma=1.0, epsilon=0.3):
    max_length = 0
    for epoch in range(MAX_EPOCHS):
        states, actions, rewards, targets = record_episode(env, q_net, epsilon, gamma)

        # Train the neural net
        episode_length = len(actions)
        states = states[:-1]  # Remove the terminal state

        batch = torch.tensor([np.append(states[i], actions[i]) for i in range(episode_length)], dtype=torch.float)
        q_net.optimizer.zero_grad()

        batch_output = q_net.forward_pass(batch)
        targets = torch.tensor(targets)

        loss = q_net.compute_loss(batch_output, targets)
        loss.backward()
        q_net.optimizer.step()
        max_length = max(max_length, episode_length)

        epsilon = max(epsilon * EPSILON_DECAY, 0.01)

        if epoch % 1000 == 999:
            print(f"Max length: {loss.item()}")

    print(f"{max_length}")

    # Train a network

    return q_net


q_net = DQN()

train_qlearn(environment, q_net)
