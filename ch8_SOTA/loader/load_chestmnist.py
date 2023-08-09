import medmnist
from medmnist import ChestMNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset

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