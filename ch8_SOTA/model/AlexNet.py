import torch
from torch import nn, optim

class AlexNet(nn.Module):
    def __init__(self, C, H, W, num_classes = 10):
        super(AlexNet, self).__init__()
        self.n_chans1 = 64
        self.conv1 = nn.Conv2d(C, self.n_chans1, kernel_size=3)
        self.act1 = nn.ReLU()
        c = self.n_chans1
        h = H - 2
        w = W - 2
        self.pool1 = nn.MaxPool2d(2)
        h = h // 2
        w = w // 2
        self.conv2 = nn.Conv2d(c, c * 3, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        c = c * 3
        h = h // 2
        w = w // 2
        self.conv3 = nn.Conv2d(c, c * 2, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        c = c * 2
        self.conv4 = nn.Conv2d(c, c * 2 // 3, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        c = c * 2 // 3
        self.conv5 = nn.Conv2d(c, c, kernel_size=1)
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2)
        h = h // 2
        w = w // 2
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(c * h * w, 1024)
        self.act6 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(1024, 512)
        self.act7 = nn.ReLU()
        self.fc8 = nn.Linear(512, num_classes)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        
        x = self.conv4(x)
        x = self.act4(x)
        
        x = self.conv5(x)
        x = self.act5(x)
        x = self.pool5(x)
        
        # 모듈화 필요>>>
        sizes = x.size()
        x = x.view(sizes[0], -1)
        # <<<모듈화 필요
        
        x = self.dropout6(x)
        x = self.fc6(x)
        x = self.act6(x)
        
        x = self.dropout7(x)
        x = self.fc7(x)
        x = self.act7(x)
        
        x = self.fc8(x)
        return x
        
        
        