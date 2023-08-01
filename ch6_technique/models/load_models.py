# ch6_techs/models/load_models.py

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


class MyCNN(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.act1 = nn.ReLU()
        h = (height - 4)
        w = (width - 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.act2 = nn.ReLU()
        h = (h - 4)
        w = (w - 4)
        self.pool2 = nn.MaxPool2d(2,2)
        h = h / 2
        w = w / 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.act3 = nn.ReLU()
        h = (h - 4)
        w = (w - 4)
        self.pool3 = nn.MaxPool2d(2,2)
        h = h / 2
        w = w / 2
        # 3x3 이미지가 됨
        self.fc4 = nn.Linear(64 * int(h) * int(w), 100)
        self.act4 = nn.ReLU()
        self.fc5 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        
        # 모듈화 필요>>>
        sizes = x.size()
        x = x.view(sizes[0], -1)
        # <<<모듈화 필요
        
        x = self.fc4(x)
        x = self.act4(x)
        x = self.fc5(x)
        return x


class Net(nn.Module):
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


class NetWidth(nn.Module):
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


class NetDropout(nn.Module):
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


class NetBatchNorm(nn.Module):
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


class NetDepth(nn.Module):
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