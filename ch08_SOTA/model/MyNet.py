import torch
from torch import nn, optim

class MyNet(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        h = height
        w = width
        self.pool1 = nn.MaxPool2d(2)
        h = h / 2
        w = w / 2
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        h = h / 2
        w = w / 2
        self.fc1 = nn.Linear(8 * int(h) * int(w), 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        
        # 모듈화 필요>>>
        sizes = out.size()
        out = out.view(sizes[0], -1)
        # <<<모듈화 필요
        
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out