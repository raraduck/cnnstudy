import torch
from torch import nn, optim
import torch.nn.functional as F

class MyNetRes(nn.Module):
    def __init__(self, C=3, H=32, W=32, n_chans1=32, O=2):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(C, n_chans1, kernel_size=3, padding=1)
        h = H
        w = W
        h = h // 2
        w = w // 2
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        h = h // 2
        w = w // 2
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2, kernel_size=3, padding=1)
        h = h // 2
        w = w // 2
        self.fc1 = nn.Linear(h * w * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, O)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)),2)
        out = F.max_pool2d(torch.relu(self.conv2(out)),2)
        out1 = out
        out = F.max_pool2d(torch.relu(self.conv3(out)) + out1, 2)
        # out = out.view(-1, 4 *4 * self.n_chans1 // 2)
        out = out.view(out.shape[0], -1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
