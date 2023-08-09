from torchvision.datasets import FashionMNIST
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