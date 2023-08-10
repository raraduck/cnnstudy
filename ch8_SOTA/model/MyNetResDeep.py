import torch
from torch import nn, optim
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x
    
    
class MyNetResDeep(nn.Module):
    def __init__(self, C=3, H=32, W=32, n_chans1=32, n_blocks=10, O=2):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(C, n_chans1, kernel_size=3, padding=1)
        h = H
        w = W
        h = h // 2
        w = w // 2
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans1)])
        )
        h = h // 2
        w = w // 2
        self.fc1 = nn.Linear(h * w * n_chans1, 32)
        self.fc2 = nn.Linear(32, O)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)),2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        # out = out.view(-1, 8 * 8 * self.n_chans1)
        out = out.view(out.shape[0], -1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    

