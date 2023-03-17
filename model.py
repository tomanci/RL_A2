import numpy as np
import os
import torch
from nn import NeuralNetwork
import torch.optim as optim
import torch.nn as nn


class DQN:

    def __init__(self, alpha=0.0001):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NeuralNetwork().to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

    def forward_pass_no_grad(self, s):
        input_data = torch.tensor(np.copy(s))
        torch.reshape(input_data, (1, 4))
        with torch.no_grad():
            return self.model(input_data)

    def forward_pass(self, x):
        return self.model(x)

    def compute_loss(self, output, target):
        return self.criterion(output, target)
