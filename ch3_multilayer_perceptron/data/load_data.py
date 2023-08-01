# ch3_multilayer_perceptron/data/load_data.py

import torch
from sklearn.datasets import load_digits

def load_data_mnist():
    digits = load_digits()
    X = digits.data
    Y = digits.target

    X = torch.FloatTensor(X)
    Y = torch.LongTensor(Y)

    return X, Y