import numpy as np
import torch
from copy import deepcopy
from model import DQN
from e_greedy import epsilon_greedy
from replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(self, use_replay_buffer=False, use_target_network=False):
        self.q_net = DQN()
        self.target_q_net = deepcopy(self.q_net)
        self.epsilon = 0.6
        self.epsilon_decay = 0.99
        self.gamma = 0.90
        self.sampling_rate = 2
        self.current_state_value = None
        self.buffer_size = int(10e5) if use_replay_buffer else 1
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.transition_buffer = []
        self.use_replay_buffer = use_replay_buffer
        self.use_target_network = use_target_network

    def precompute_current_state_value(self, current_state):
        self.current_state_value = self.q_net.forward_pass_no_grad(current_state).numpy()

    def select_action_for_state(self, state):
        if self.use_target_network:
            q_sa = self.target_q_net.forward_pass_no_grad(state).numpy()
        else:
            q_sa = self.q_net.forward_pass_no_grad(state).numpy()

        return epsilon_greedy(q_sa, epsilon=self.epsilon)

    def get_state_q_value(self, state):
        return self.q_net.forward_pass_no_grad(state).numpy()

    def calculate_target(self, transition):
        next_state_value = self.get_state_q_value(transition.next_state)
        current_state_value = self.get_state_q_value(transition.state)

        if transition.terminated:
            target = transition.reward
        else:
            target = transition.reward + self.gamma * max(next_state_value)

        current_state_value[transition.action] = target
        return current_state_value

    def train_agent(self, flush_buffer=False):
        if self.use_replay_buffer and self.replay_buffer.size() < self.sampling_rate:
            print("[WARNING]: Buffer size is less than the sampling rate")
            return

        if self.use_replay_buffer:
            sample_transitions = self.replay_buffer.get_n_samples(self.sampling_rate)
            self.train_with_transitions(sample_transitions)
        elif len(self.transition_buffer) == self.sampling_rate:
            self.train_with_transitions(self.transition_buffer)
            self.transition_buffer = []
        elif flush_buffer and len(self.transition_buffer) != 0:
            self.train_with_transitions(self.transition_buffer)
            self.transition_buffer = []

    def train_with_transitions(self, transitions):
        batch_states = []
        batch_targets = []

        for transition in transitions:
            target = self.calculate_target(transition)
            batch_states.append(transition.state)
            batch_targets.append(target)

        self.train_with_batch(batch_states, batch_targets)

    def train_with_batch(self, input_batch, targets):
        model_input = torch.tensor(input_batch)
        model_targets = torch.tensor(targets)

        self.q_net.optimizer.zero_grad()
        model_output = self.q_net.forward_pass(model_input)

        loss = self.q_net.compute_loss(model_output, model_targets)
        loss.backward()
        self.q_net.optimizer.step()

        return loss.item()

    def on_epoch_ended(self):
        self.train_agent(flush_buffer=True)
        self.decay_epsilon()
        if self.use_target_network:
            self.target_q_net = deepcopy(self.q_net)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def store_transition(self, transition):
        if self.use_replay_buffer:
            self.replay_buffer.add_transition(transition)
        else:
            self.transition_buffer.append(transition)