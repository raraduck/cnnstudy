
# ch7_medmnist/loader/data_loader.py

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
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



def load_cifar100(batch_size=256):
    train = CIFAR100(root="./_data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = CIFAR100(root="./_data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    classnames={
        "0": "apple",
        "1": "aquarium_fish",
        "2": "baby",
        "3": "bear",
        "4": "beaver",
        "5": "bed",
        "6": "bee",
        "7": "beetle",
        "8": "bicycle",
        "9": "bottle",
        "10": "bowl",
        "11": "boy",
        "12": "bridge",
        "13": "bus",
        "14": "butterfly",
        "15": "camel",
        "16": "can",
        "17": "castle",
        "18": "caterpillar",
        "19": "cattle",
        "20": "chair",
        "21": "chimpanzee",
        "22": "clock",
        "23": "cloud",
        "24": "cockroach",
        "25": "couch",
        "26": "cra",
        "27": "crocodile",
        "28": "cup",
        "29": "dinosaur",
        "30": "dolphin",
        "31": "elephant",
        "32": "flatfish",
        "33": "forest",
        "34": "fox",
        "35": "girl",
        "36": "hamster",
        "37": "house",
        "38": "kangaroo",
        "39": "keyboard",
        "40": "lamp",
        "41": "lawn_mower",
        "42": "leopard",
        "43": "lion",
        "44": "lizard",
        "45": "lobster",
        "46": "man",
        "47": "maple_tree",
        "48": "motorcycle",
        "49": "mountain",
        "50": "mouse",
        "51": "mushroom",
        "52": "oak_tree",
        "53": "orange",
        "54": "orchid",
        "55": "otter",
        "56": "palm_tree",
        "57": "pear",
        "58": "pickup_truck",
        "59": "pine_tree",
        "60": "plain",
        "61": "plate",
        "62": "poppy",
        "63": "porcupine",
        "64": "possum",
        "65": "rabbit",
        "66": "raccoon",
        "67": "ray",
        "68": "road",
        "69": "rocket",
        "70": "rose",
        "71": "sea",
        "72": "seal",
        "73": "shark",
        "74": "shrew",
        "75": "skunk",
        "76": "skyscraper",
        "77": "snail",
        "78": "snake",
        "79": "spider",
        "80": "squirrel",
        "81": "streetcar",
        "82": "sunflower",
        "83": "sweet_pepper",
        "84": "table",
        "85": "tank",
        "86": "telephone",
        "87": "television",
        "88": "tiger",
        "89": "tractor",
        "90": "train",
        "91": "trout",
        "92": "tulip",
        "93": "turtle",
        "94": "wardrobe",
        "95": "whale",
        "96": "willow_tree",
        "97": "wolf",
        "98": "woman",
        "99": "worm",
    }
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, classnames