# ch4_dataloader/data/load_data.py

import torch
from torch import nn, optim


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__()
        self.ln = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.ln(x)
        return x



class MyMLP(nn.Module):
    def __init__(self, in_features=64, out_features=10):
        super().__init__()
        self.ln1 = nn.Linear(in_features, 32)
        self.relu1 = nn.ReLU()
        self.ln2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.ln3 = nn.Linear(16, out_features)
        # self.relu = nn.ReLU

    def forward(self, x):
        sizes = x.size()
        x = x.view(sizes[0], -1)
        x = self.ln1(x)
        x = self.relu1(x) # x = torch.relu(x)
        x = self.ln2(x)
        x = self.relu2(x) # x = torch.relu(x)
        x = self.ln3(x)
        return x
        


class MyCNN(nn.Module):
    def __init__(self, batch_size):
        super(MyCNN, self).__init__()
        self.batch_size = batch_size
        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(16,32, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(self.batch_size,-1)
        out = self.fc_layer(out)
        return out