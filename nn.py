import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
