import torch
from torch import nn, optim

class MyNetBatchNorm(nn.Module):
    def __init__(self, height, width, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(1, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.act1 = nn.Tanh()
        h = height
        w = width
        self.pool1 = nn.MaxPool2d(2)
        h = h / 2
        w = w / 2
        
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 2)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        h = h / 2
        w = w / 2
        
        self.fc1 = nn.Linear(int(h) * int(w) * n_chans1 // 2, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        # out = F.max_pool2d(torch.tanh(out), 2)
        out = self.pool1(self.act1(out))
        
        out = self.conv2_batchnorm(self.conv2(out))
        # out = F.max_pool2d(torch.tanh(out), 2)
        out = self.pool2(self.act2(out))
                         
        # 모듈화 필요>>>
        out = out.view(out.shape[0], -1)
        # <<<모듈화 필요
        
        # out = torch.tanh(self.fc1(out))
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out