import gymnasium as gym
import numpy as np
import argparse
from tqdm import tqdm
from numpy import asarray, save, load

from transition import Transition
from agent import DQNAgent
from config import Config, load_from_yaml
from e_greedy import EpsilonGreedy
from boltzmann import Boltzmann


def train_agent_on_env(agent, env, n_epochs):
    for epoch in tqdm(range(n_epochs)):
        reward = perform_an_episode(agent, env)
        yield reward


def perform_an_episode(agent, env) -> int:
    state, _ = env.reset()
    terminated = False

    total_reward: int = 0

    while not terminated:
        action = agent.select_action_for_state(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(Transition(state, action, reward, next_state, terminated))
        agent.train_agent()

        total_reward += int(reward)

        state = next_state

        if truncated:
            break

    agent.on_epoch_ended()

    return total_reward


def read_cli_args():
    parser = argparse.ArgumentParser(description="My parser")
    parser.add_argument('-er', '--experience-replay', dest='use_experience_replay', action='store_true')
    parser.add_argument('-tn', '--target-network', dest='use_target_network', action='store_true')
    parser.add_argument('-f', '--config-file', dest='config_file', type=str, default="")
    args = parser.parse_args()
    return args


def run(config: Config, experience_replay: bool, target_network: bool):
    environment = gym.make("CartPole-v1")
    print(f"Starting DQN, experience-replay: {experience_replay}, target-network: {target_network}")

    match config.policy:
        case "epsilon_greedy":
            action_selection_policy = EpsilonGreedy(config.epsilon)
        case "boltzmann":
            action_selection_policy = Boltzmann(config.temp)
        case policy:
            raise NotImplementedError(f"Policy '{policy}' is not implemented")

    agent = DQNAgent(
        config=config,
        action_selection_policy=action_selection_policy,
        use_replay_buffer=experience_replay,
        use_target_network=target_network
    )

    rewards = list(train_agent_on_env(agent, environment, config.epochs))
    print(f"Average reward: {np.mean(rewards)}, average over last 100 epochs: {np.mean(rewards[-100:])}")
    return rewards


def perform_experiment(config_file_path, config, use_experience_replay, use_target_network):
    rewards = np.ndarray((3, config.epochs), dtype=int)
    for i in range(3):
        experiment_rewards = run(config, use_experience_replay, use_target_network)
        rewards[i] = experiment_rewards

    data = asarray(rewards)
    file_name = 'configs/' + config_file_path.split('.')[0]
    save(file_name, data)


def main():
    args = read_cli_args()
    config_file_path = args.config_file
    config = load_from_yaml(config_file_path) if config_file_path != "" else Config()

    # run(config, args.use_experience_replay, args.use_target_network)
    perform_experiment(config_file_path, config, args.use_experience_replay, args.use_target_network)


if __name__ == "__main__":
    main()
