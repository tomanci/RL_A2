import gymnasium as gym
import numpy as np
from transition import Transition
from agent import DQNAgent

environment = gym.make("CartPole-v1")
use_experience_replay = True

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

    agent.train_agent(flush_buffer=True)
    agent.decay_epsilon()

    return rewards


train_agent_on_env(DQNAgent(use_replay_buffer=use_experience_replay), environment)
