import torch
from torch import nn, optim

class MyNetDepth(nn.Module):
    def __init__(self, height, width, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(1, n_chans1, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        h = height
        w = width
        self.pool1 = nn.MaxPool2d(2)
        h = h // 2
        w = w // 2
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        h = h // 2
        w = w // 2
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        h = h // 2
        w = w // 2
        self.fc1 = nn.Linear(h * w * n_chans1 // 2, 32)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = self.pool3(self.act3(self.conv3(out)))
        
        # 모듈화 필요>>>
        # sizes = out.size()
        # out = out.view(sizes[0], -1)
        out = out.view(out.shape[0], -1)
        # <<<모듈화 필요
        
        out = self.act4(self.fc1(out))
        out = self.fc2(out)
        return out
