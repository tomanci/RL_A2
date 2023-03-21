import numpy as np
import torch
from copy import deepcopy
from model import DQN
from e_greedy import epsilon_greedy


class DQNAgent:

    def __init__(self):
        self.q_net = DQN()
        self.epsilon = 0.6
        self.epsilon_decay = 0.99
        self.gamma = 0.90
        self.current_state_value = None

    def precompute_current_state_value(self, current_state):
        self.current_state_value = self.q_net.forward_pass_no_grad(current_state).numpy()

    def select_action_for_state(self, state):
        q_sa = self.q_net.forward_pass_no_grad(state).numpy()
        return epsilon_greedy(q_sa, epsilon=self.epsilon)

    def get_state_q_value(self, state):
        return self.q_net.forward_pass_no_grad(state).numpy()

    def calculate_target(self, state, action, next_state, reward, terminated):
        next_state_value = self.q_net.forward_pass_no_grad(next_state)
        current_state_value = self.get_state_q_value(state)

        if terminated:
            target = reward
        else:
            target = reward + self.gamma * torch.max(next_state_value).item()

        current_state_value[action] = target
        return current_state_value

    def train_with_batch(self, input_batch, targets):
        model_input = torch.tensor(input_batch)
        model_targets = torch.tensor(targets)

        self.q_net.optimizer.zero_grad()
        model_output = self.q_net.forward_pass(model_input)

        loss = self.q_net.compute_loss(model_output, model_targets)
        loss.backward()
        self.q_net.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
