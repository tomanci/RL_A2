from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.linear_relu_stack = nn.Sequential()
        for i in range(0, len(architecture) - 2, 2):
            self.linear_relu_stack.append(nn.Linear(architecture[i], architecture[i + 1]))
            self.linear_relu_stack.append(nn.ReLU())

        last_element_index = len(architecture) - 1
        self.linear_relu_stack.append(nn.Linear(architecture[last_element_index - 1], architecture[last_element_index]))

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
