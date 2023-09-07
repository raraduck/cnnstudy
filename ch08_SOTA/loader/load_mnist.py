from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset

def load_mnist(batch_size=256):
    mnist_train = MNIST("./_data", train=True, download=True,
                        transform=transforms.ToTensor())
    mnist_test = MNIST("./_data", train=False, download=True,
                        transform=transforms.ToTensor())
    classnames = {
        0:'(0)',
        1:'(1)',
        2:'(2)',
        3:'(3)',
        4:'(4)',
        5:'(5)',
        6:'(6)',
        7:'(7)',
        8:'(8)',
        9:'(9)',
    }
    m_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    m_test_laoder = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return m_train_loader, m_test_laoder, classnames