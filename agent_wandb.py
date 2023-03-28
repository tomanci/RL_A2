
from typing import List
import numpy as np

import wandb
from config import Config

from action_slection import ActionSlection
from agent import DQNAgent


class WandbDQNAgent(DQNAgent):

    def __init__(self, config: Config, action_selection_policy: ActionSlection, use_replay_buffer=False, use_target_network=False):
        super().__init__(config, action_selection_policy, use_replay_buffer, use_target_network)
        self.run_rewards: List[int] = []

        # Log Pytorch model to wandb
        wandb.watch(self.q_net.model.linear_relu_stack, log="all", log_freq=100, log_graph=True)
 

    def on_epoch_ended(self, total_reward: int):
        self.run_rewards.append(total_reward)
        avg_reward_last_100 = np.mean(self.run_rewards[-100:])
        metrics={"episode_reward": total_reward, "avg_reward_last_100": avg_reward_last_100}
        wandb.log(metrics)
        return super().on_epoch_ended(total_reward)

