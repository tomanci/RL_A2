import gymnasium as gym
import numpy as np
from transition import Transition
from agent import DQNAgent
import argparse


use_experience_replay = False
use_target_network = False

parser = argparse.ArgumentParser(description="My parser")
parser.add_argument('--experience-replay', dest='use_experience_replay', action='store_true')
parser.add_argument('--target-network', dest='use_target_network', action='store_true')
args = parser.parse_args()

environment = gym.make("CartPole-v1")

experience_replay = args.use_experience_replay
target_network = args.use_target_network

print(f"Starting DQN, experience-replay: {experience_replay}, target-network: {target_network}")

MAX_EPOCHS = 1000


def train_agent_on_env(agent, env):
    epoch_rewards = []

    for epoch in range(MAX_EPOCHS):
        rewards = perform_an_episode(agent, env)
        epoch_rewards.append(rewards)

    print(f"Average reward: {np.mean(epoch_rewards)}")


def perform_an_episode(agent, env):
    state, _ = env.reset()
    terminated = False

    rewards = 0

    while not terminated:
        action = agent.select_action_for_state(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(Transition(state, action, reward, next_state, terminated))
        agent.train_agent()

        rewards += reward

        state = next_state

        if truncated:
            break

    agent.on_epoch_ended()

    return rewards


train_agent_on_env(DQNAgent(use_replay_buffer=experience_replay, use_target_network=target_network), environment)
