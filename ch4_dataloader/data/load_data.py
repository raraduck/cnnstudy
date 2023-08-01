# ch4_dataloader/data/load_data.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits

def load_data_mnist():
    digits = load_digits()
    X = digits.data
    Y = digits.target

    X = torch.FloatTensor(X)
    Y = torch.LongTensor(Y)

    return X, Y


def loader_data_mnist_sklearn(batch_size = 256):
    digits = fetch_openml('MNIST_784', version=1, data_home='./data/')
    # digits = load_digits()
    X = digits.data
    Y = digits.target

    X = torch.reshape(torch.FloatTensor(X.values) / 255, (-1, 1, 28, 28))
    Y = torch.LongTensor([int(x) for x in Y])

    n_samples = X.shape[0]
    n_val = int(0.2 * n_samples)
    shuffled_indices = torch.randperm(n_samples)
    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]
    X_train = X[train_indices,:]
    Y_train = Y[train_indices]
    X_val = X[val_indices,:]
    Y_val = Y[val_indices]

    ds_train = TensorDataset(X_train, Y_train)
    ds_val = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader
    


def loader_data_mnist_torchvision(batch_size = 256):
    mnist_train = dset.MNIST("./data/", train=True, transform=transforms.ToTensor(),
                             target_transform=None, download=True)
    mnist_test = dset.MNIST("./data/", train=False, transform=transforms.ToTensor(),
                            target_transform=None, download=True)
    
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                               shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size,
                                              shuffle=True, num_workers=0, drop_last=True)

    return train_loader, test_loader