import torch
from torch import nn, optim

class MyNetDropout(nn.Module):
    def __init__(self, height, width, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(1, n_chans1, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        h = height
        w = width
        self.pool1 = nn.MaxPool2d(2)
        h = h / 2
        w = w / 2
        self.conv1_dropout = nn.Dropout2d(p=0.4)
        
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        h = h
        w = w
        self.pool2 = nn.MaxPool2d(2)
        h = h / 2
        w = w / 2
        self.conv2_dropout = nn.Dropout2d(p=0.4)
        
        self.fc1 = nn.Linear(int(h) * int(w) * n_chans1 // 2, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.conv1_dropout(out)
        
        out = self.pool2(self.act2(self.conv2(out)))
        out = self.conv2_dropout(out)
        
        # 모듈화 필요>>>
        sizes = out.size()
        out = out.view(sizes[0], -1)
        # <<<모듈화 필요
        
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out