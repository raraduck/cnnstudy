import torch
from torch import nn, optim

class MyNetWidth(nn.Module):
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
        
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        h = h / 2
        w = w / 2
        
        self.fc1 = nn.Linear(int(h) * int(w) * n_chans1 // 2, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        # out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = self.pool1(self.act1(self.conv1(x)))
        # out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = self.pool2(self.act2(self.conv2(out)))

        # 모듈화 필요>>>
        # out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        sizes = out.size()
        out = out.view(sizes[0], -1)
        # <<<모듈화 필요
        
        # out = torch.tanh(self.fc1(out))
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out