
# ch7_medmnist/loader/data_loader.py

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset


def load_fashionmnist(batch_size=256, trformers=None):
    if trformers==None:
        fmnist_train = FashionMNIST("./_data", train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))
        fmnist_test = FashionMNIST("./_data", train=False, download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))   
    else:
        fmnist_train = FashionMNIST("./_data", train=True, download=True,
                                   transform=trformers) 
        fmnist_test = FashionMNIST("./_data", train=False, download=True,
                                   transform=trformers) 
        
    classnames = {
        0:'T-shirt/top',
        1:'Trouser',
        2:'Pullover',
        3:'Dress',
        4:'Coat',
        5:'Sandal',
        6:'Shirt',
        7:'Sneaker',
        8:'Bag',
        9:'Ankle boot',
    }
    f_train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    f_test_laoder = DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, drop_last=True)

    return f_train_loader, f_test_laoder, classnames


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
    
    return m_train_loader, m_test_laoder


def load_pneumoniamnist(batch_size=256):
    import medmnist
    from medmnist import PneumoniaMNIST
    
    train = PneumoniaMNIST(root="./_data", split="train", download=True, 
                           transform=transforms.Compose([transforms.ToTensor()]))
    test = PneumoniaMNIST(root="./_data", split="test", download=True, 
                          transform=transforms.Compose([transforms.ToTensor()]))
    classnames={
        0: 'normal', 
        1: 'pneumonia'
    }
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, classnames


def load_chestmnist(batch_size=256):
    import medmnist
    from medmnist import ChestMNIST
    
    train = ChestMNIST(root="./_data", split="train", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = ChestMNIST(root="./_data", split="test", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    classnames={
        0: 'atelectasis', 
        1: 'cardiomegaly', 
        2: 'effusion', 
        3: 'infiltration', 
        4: 'mass', 
        5: 'nodule', 
        6: 'pneumonia', 
        7: 'pneumothorax', 
        8: 'consolidation', 
        9: 'edema', 
        10: 'emphysema', 
        11: 'fibrosis', 
        12: 'pleural', 
        13: 'hernia'
    }
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, classnames



def load_cifar10(batch_size=256):
    train = CIFAR10(root="./_data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = CIFAR10(root="./_data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    classnames={
        '0': 'airplane', 
        '1': 'automobile', 
        '2': 'bird', 
        '3': 'cat', 
        '4': 'deer', 
        '5': 'dog', 
        '6': 'frog', 
        '7': 'horse', 
        '8': 'ship', 
        '9': 'truck', 
    }
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, classnames