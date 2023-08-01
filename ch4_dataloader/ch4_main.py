# ch4_dataloader/ch4_main.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from data.load_data import loader_data_mnist_sklearn, loader_data_mnist_torchvision

batch_size = 256
learning_rate = 0.0002
num_epoch = 10

def main(batch_size):
    from data.load_data import loader_data_mnist_torchvision
    train_loader, test_loader = loader_data_mnist_torchvision()
    print(type(train_loader))
    print(train_loader.dataset[0][0].shape)
    
    from models.load_models import MyMLP, MyCNN
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sizes = train_loader.dataset[0][0].shape
    in_features = train_loader.dataset[0][0].view(sizes[0], -1).shape[-1]
    model_MyMLP = MyMLP(in_features=in_features, out_features=10).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_MyMLP.parameters(), lr=learning_rate)

    import collections
    from train.run_training import training_loop, validate
    training_loop(10, optimizer, model_MyMLP, loss_func, train_loader, device=device)
    all_acc_dict = collections.OrderedDict()
    all_acc_dict["mymlp"] = validate(model_MyMLP, train_loader, test_loader, device=device)


    model_MyCNN = MyCNN(batch_size=batch_size).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_MyCNN.parameters(), lr=learning_rate)

    training_loop(10, optimizer, model_MyCNN, loss_func, train_loader, device=device)
    all_acc_dict["mycnn"] = validate(model_MyCNN, train_loader, test_loader, device=device)

    from utils.load_utils import compare_validations
    compare_validations(all_acc_dict)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hyper-parameters')
    parser.add_argument('--batch_size', metavar='val',
                        help='batch_size', default=256)
    # parser.add_argument('--epoc', metavar='N',
    #                     help='num of epoc', default=100)
    args = parser.parse_args()
    main(batch_size=args.batch_size)
