# ch5_CNN/ch5_main.py
from data.load_data import load_mnist_torchvision, load_fashion_mnist_torchvision
from models.load_models import MyCNN
from train.run_training import training_loop, validate

import torch
from torch import nn, optim
def mycnn_MNIST(batch_size):
    mnist_train_loader, mnist_test_loader = load_mnist_torchvision(
        target_path="./data/MNIST",
        batch_size=256)
    
    learning_rate = 0.0002
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    training_loop(10, optimizer, model, loss_fn, mnist_train_loader, mnist_test_loader, device=device)
    return validate(model, mnist_train_loader, mnist_test_loader, device=device)



    
def mycnn_FashionMNIST(batch_size):
    train_loader, test_loader = load_fashion_mnist_torchvision(
        target_path="./data/FashionMNIST",
        batch_size=256)
    
    learning_rate = 0.0002
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    training_loop(10, optimizer, model, loss_fn, train_loader, test_loader, device=device)
    return validate(model, train_loader, test_loader, device=device)


    
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hyper-parameters')
    parser.add_argument('--batch_size', metavar='val', help='batch_size', default=256)
    args = parser.parse_args()

    
    import collections
    all_acc_dict = collections.OrderedDict()
    all_acc_dict["mnist"] = mycnn_MNIST(batch_size=args.batch_size)
    all_acc_dict["fmnist"] = mycnn_FashionMNIST(batch_size=args.batch_size)


    from utils.load_utils import compare_validations
    compare_validations(all_acc_dict)