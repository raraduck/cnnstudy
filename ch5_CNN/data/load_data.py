# ch5_CNN/data/load_data.py

def load_fashion_mnist_torchvision(target_path="./data", batch_size = 256):
    from torchvision.datasets import FashionMNIST
    from torchvision import transforms
    fashion_mnist_train = FashionMNIST(target_path,
                                   train=True, download=True,
                                   transform=transforms.ToTensor())
    fashion_mnist_test = FashionMNIST(target_path,
                                   train=False, download=True,
                                   transform=transforms.ToTensor())
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    
    train_loader = DataLoader(fashion_mnist_train,
                              batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(fashion_mnist_test,
                             batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


def load_mnist_torchvision(target_path="./data", batch_size = 256):
    from torchvision.datasets import MNIST
    from torchvision import transforms
    mnist_train = MNIST(target_path, 
                        train=True, download=True,
                        transform=transforms.ToTensor())
    mnist_test = MNIST(target_path, 
                        train=True, download=True,
                        transform=transforms.ToTensor())
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    
    train_loader = DataLoader(mnist_train,
                              batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test,
                             batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader

