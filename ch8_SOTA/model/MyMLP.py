import torch
from torch import nn, optim


class MyMLP(nn.Module):
    def __init__(self, in_features=64, out_features=10):
        super().__init__()
        self.ln1 = nn.Linear(in_features, 32)
        self.relu1 = nn.ReLU()
        self.ln2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.ln3 = nn.Linear(16, out_features)

    def forward(self, x):
        sizes = x.size()
        x = x.view(sizes[0], -1)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.ln3(x)
        return x