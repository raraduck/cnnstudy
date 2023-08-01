# ch2_logistic_regression/models/load_models.py

from torch import nn, optim

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__()
        self.ln = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.ln(x)
        return x