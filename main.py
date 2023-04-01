import gymnasium as gym
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from random_policy import RandomPolicy

from transition import Transition
from agent import DQNAgent
from config import Config, load_from_yaml
from e_greedy import EpsilonGreedy
from boltzmann import Boltzmann


def train_agent_on_env(agent, env, n_epochs):
    for epoch in (pbar := tqdm(range(n_epochs), desc="Epochs", position=0)):
        reward = perform_an_episode(agent, env)
        pbar.set_postfix({"last_reward": reward})
        yield reward


def perform_an_episode(agent: DQNAgent, env) -> int:
    state, _ = env.reset()

    total_reward: int = 0


    for step in tqdm(list(range(500)), desc="Environment steps", position=1, leave=False): # maximum number of steps. May terminate earlier
        action = agent.select_action_for_state(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(Transition(state, action, reward, next_state, terminated))
        agent.train_agent()

        total_reward += int(reward)

        state = next_state

        if terminated or truncated:
            break

    agent.on_epoch_ended(total_reward)

    return total_reward


def read_cli_args():
    parser = argparse.ArgumentParser(description="My parser")
    parser.add_argument('-er', '--experience-replay', dest='use_experience_replay', action='store_true')
    parser.add_argument('-tn', '--target-network', dest='use_target_network', action='store_true')
    parser.add_argument('-f', '--config-file', dest='config_file', type=str, default="")
    parser.add_argument('-d', '--config-file-dir', dest='config_file_dir', type=str, default="")
    parser.add_argument('-r', '--repetitions', dest='repetitions', type=int, default=5)
    args = parser.parse_args()

    if args.config_file and args.config_file_dir:
        raise Exception("Please provide either a single config file or a directory with config files")

    return args


def run(config: Config, experience_replay: bool, target_network: bool, agent_class=DQNAgent):
    environment = gym.make("CartPole-v1")
    print(f"Starting DQN, experience-replay: {experience_replay}, target-network: {target_network}")

    match config.policy:
        case "epsilon_greedy":
            action_selection_policy = EpsilonGreedy(config.epsilon)
        case "boltzmann":
            action_selection_policy = Boltzmann(config.temp)
        case "random":
            action_selection_policy = RandomPolicy()
        case policy:
            raise NotImplementedError(f"Policy '{policy}' is not implemented")

    agent = agent_class(
        config=config,
        action_selection_policy=action_selection_policy,
        use_replay_buffer=experience_replay,
        use_target_network=target_network
    )

    rewards = list(train_agent_on_env(agent, environment, config.epochs))
    print(f"Average reward: {np.mean(rewards)}, average over last 100 epochs: {np.mean(rewards[-100:])}")
    return rewards


def perform_experiment(config_file_path: str, config: Config, use_experience_replay: bool, use_target_network: bool, repetitions=5):
    rewards = np.ndarray((repetitions, config.epochs), dtype=int)
    for i in range(repetitions):
        experiment_rewards = run(config, use_experience_replay, use_target_network)
        rewards[i] = experiment_rewards

    data = np.asarray(rewards)
    file_name = config_file_path.replace("configs/", "results/").replace(".yaml", ".npy")
    print("Saving experiment results", file_name)
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    np.save(file_name, data)


def main():
    args = read_cli_args()
    config_file_dir: str = args.config_file_dir
    repetitions: int = args.repetitions
    if config_file_dir:

        if not Path(config_file_dir).exists():
            raise Exception("The provided config file directory does not exist")
        else:
            print("Loading configs from directory", config_file_dir)

        for config_file_path in Path(config_file_dir).glob("*.yaml"):
            config_file_path = str(config_file_path)
            config = load_from_yaml(config_file_path)
            perform_experiment(config_file_path, config, args.use_experience_replay, args.use_target_network, repetitions=repetitions)
    else:
        config_file_path: str = args.config_file
        config = load_from_yaml(config_file_path) if config_file_path != "" else Config()

        # run(config, args.use_experience_replay, args.use_target_network)
        perform_experiment(config_file_path, config, args.use_experience_replay, args.use_target_network, repetitions=repetitions)

if __name__ == "__main__":
    main()
