import gymnasium as gym
import numpy as np
from agent import DQNAgent

environment = gym.make("CartPole-v1")

MAX_EPOCHS = 1000


def train_agent_on_env(agent, env):
    epoch_rewards = []

    for epoch in range(MAX_EPOCHS):
        rewards = perform_an_episode(agent, env)
        print(f"Rewards: {rewards}")
        epoch_rewards.append(rewards)

    print(f"Average reward: {np.mean(epoch_rewards)}")


def perform_an_episode(agent, env):
    state, _ = env.reset()
    terminated = False

    batch_size = 2
    rewards = 0

    batch_states = []
    batch_targets = []

    while not terminated:
        action = agent.select_action_for_state(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        target_q_value = agent.calculate_target(state, action, next_state, reward, terminated)

        batch_states.append(state)
        batch_targets.append(target_q_value)

        rewards += reward

        state = next_state

        # Train agent
        if len(batch_states) == batch_size:
            agent.train_with_batch(batch_states, batch_targets)
            batch_states, batch_targets = [], []

        if truncated:
            break

    if len(batch_states) != 0:
        agent.train_with_batch(batch_states, batch_targets)

    agent.decay_epsilon()

    return rewards


train_agent_on_env(DQNAgent(), environment)
