import torch
from sklearn.datasets import load_iris, load_digits

def load_data_iris():
    iris = load_iris()
    X = iris.data[:100]
    y = iris.target[:100]
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    return X_t, y_t


def load_data_mnist():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    return X, y

