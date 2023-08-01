# ch6_techs/data/load_data.py

from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset


def load_fashion_mnist(batch_size=256):
    fmnist_train = FashionMNIST("./data", train=True, download=True,
                               transform=transforms.ToTensor())
    fmnist_test = FashionMNIST("./data", train=False, download=True,
                               transform=transforms.ToTensor())   
    
    f_train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    f_test_laoder = DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, drop_last=True)

    return f_train_loader, f_test_laoder


def load_mnist(batch_size=256):
    mnist_train = MNIST("./data", train=True, download=True,
                        transform=transforms.ToTensor())
    mnist_test = MNIST("./data", train=False, download=True,
                        transform=transforms.ToTensor())
    
    m_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    m_test_laoder = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return m_train_loader, m_test_laoder