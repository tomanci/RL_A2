from typing import List
import numpy as np
import torch
from copy import deepcopy
from config import Config
from model import DQN
from replay_buffer import ReplayBuffer
from transition import Transition
from action_slection import ActionSlection



class DQNAgent:

    def __init__(self, config: Config, action_selection_policy: ActionSlection,  use_replay_buffer=False, use_target_network=False):
        self.config = config
        self.q_net = DQN(nn_architecture=self.config.nn_architecture, alpha=self.config.learning_rate)
        self.target_q_net = deepcopy(self.q_net)
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.transition_buffer: List[Transition] = []
        self.use_replay_buffer = use_replay_buffer
        self.use_target_network = use_target_network
        self.target_network_sync_counter = 0
        self.target_network_sync_freq = self.config.target_network_sync_freq

        self.action_selection_policy = action_selection_policy
        
    def select_action_for_state(self, state):
        q_sa = self.q_net.forward_pass_no_grad(state).numpy()
        return self.action_selection_policy.select_action(q_sa)

    def get_state_q_value(self, state):
        return self.q_net.forward_pass_no_grad(state).numpy()

    def calculate_target(self, transition: Transition):
        next_state_value = self.get_state_q_value(transition.next_state)
        current_state_value = self.get_state_q_value(transition.state)

        if transition.terminated:
            target = transition.reward
        else:
            target = transition.reward + self.config.gamma * max(next_state_value)

        current_state_value[transition.action] = target
        return current_state_value

    def train_agent(self, flush_buffer=False):
        if self.use_replay_buffer and self.replay_buffer.size() < self.config.sampling_rate:
            # print("[WARNING]: Current buffer size is less than the sampling rate. Skipping this training step until more transitions are in the buffer.")
            return

        if self.use_replay_buffer:
            sample_transitions = self.replay_buffer.get_n_samples(self.config.sampling_rate)
            self.train_with_transitions(sample_transitions)
        elif len(self.transition_buffer) == self.config.sampling_rate:
            self.train_with_transitions(self.transition_buffer)
            self.transition_buffer = []
        elif flush_buffer and len(self.transition_buffer) != 0:
            self.train_with_transitions(self.transition_buffer)
            self.transition_buffer = []

    def train_with_transitions(self, transitions: List[Transition]):
        batch_states = []
        batch_targets = []

        for transition in transitions:
            target = self.calculate_target(transition)
            batch_states.append(transition.state)
            batch_targets.append(target)

        self.train_with_batch(np.array(batch_states), np.array(batch_targets))

    def train_with_batch(self, input_batch: np.ndarray, targets: np.ndarray):
        model_input = torch.tensor(input_batch)
        model_targets = torch.tensor(targets)

        self.q_net.optimizer.zero_grad()
        model_output = self.q_net.forward_pass(model_input)

        loss = self.q_net.compute_loss(model_output, model_targets)
        loss.backward()
        self.q_net.optimizer.step()

        return loss.item()

    def on_epoch_ended(self, total_reward: int):
        self.train_agent(flush_buffer=True)
        self.action_selection_policy.decay()
        if self.use_target_network:
            if self.target_network_sync_counter == self.target_network_sync_freq:
                self.target_q_net = deepcopy(self.q_net)
                self.target_network_sync_counter = 0
            else:
                self.target_network_sync_counter += 1

        self.q_net.scheduler.step() # decay learning rate after each epoch

    def store_transition(self, transition):
        if self.use_replay_buffer:
            self.replay_buffer.add_transition(transition)
        else:
            self.transition_buffer.append(transition)
