from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset

def load_cifar10(batch_size=256, trforms=None):
    if trforms==None:
        train = CIFAR10(root="./_data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test = CIFAR10(root="./_data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    else:
        train = CIFAR10(root="./_data", train=True, download=True, transform=trforms) 
        test = CIFAR10(root="./_data", train=False, download=True, transform=trforms) 
        

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