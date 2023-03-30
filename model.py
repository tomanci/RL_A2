import numpy as np
import torch
from nn import NeuralNetwork
import torch.optim as optim
import torch.nn as nn


class DQN:

    def __init__(self, nn_architecture, alpha=0.0001, learning_rate_decay=1.0):
        if torch.cuda.is_available():
            print("CUDA device detected but this wont be used due to small network size.")
        self.device = "cpu"
        print(f"Device: {self.device}")
        self.model = NeuralNetwork(nn_architecture).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=learning_rate_decay)

    def forward_pass_no_grad(self, s):
        input_data = torch.tensor(np.copy(s)).to(self.device)
        torch.reshape(input_data, (1, 4))
        with torch.no_grad():
            return self.model(input_data).cpu()

    def forward_pass(self, x):
        return self.model(x.to(self.device)).cpu()

    def compute_loss(self, output, target):
        # return (target - output).pow(2).sum()
        return self.criterion(output, target)
