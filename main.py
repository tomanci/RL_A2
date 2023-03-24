import gymnasium as gym
import numpy as np
from transition import Transition
from agent import DQNAgent
from config import defaultConfig, load_from_yaml
import argparse

parser = argparse.ArgumentParser(description="My parser")
parser.add_argument('-er', '--experience-replay', dest='use_experience_replay', action='store_true')
parser.add_argument('-tn', '--target-network', dest='use_target_network', action='store_true')
parser.add_argument('-f', '--config-file', dest='config_file', type=str, default="")
args = parser.parse_args()

config_file_path = args.config_file
config = load_from_yaml(config_file_path) if config_file_path != "" else defaultConfig

environment = gym.make("CartPole-v1")

experience_replay = args.use_experience_replay
target_network = args.use_target_network

print(f"Starting DQN, experience-replay: {experience_replay}, target-network: {target_network}, config-file: {config_file_path}")

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


agent = DQNAgent(
    config=config,
    use_replay_buffer=experience_replay,
    use_target_network=target_network
)

train_agent_on_env(agent, environment)
