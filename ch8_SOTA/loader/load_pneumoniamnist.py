import medmnist
from medmnist import PneumoniaMNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset

def load_pneumoniamnist(batch_size=256):
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