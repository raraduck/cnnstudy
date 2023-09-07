# ch7_medmnist/main.py

# from data.load_data import load_mnist_torchvision, load_fashion_mnist_torchvision
# from models.load_models import MyCNN
# from train.run_training import training_loop, validate

import torch
from torch import nn, optim
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hyper-parameters')
    parser.add_argument('--batch_size', metavar='val', help='batch_size', default=256)
    args = parser.parse_args()

    print('hello world')
    
    # import collections
    # all_acc_dict = collections.OrderedDict()
    # all_acc_dict["mnist"] = mycnn_MNIST(batch_size=args.batch_size)
    # all_acc_dict["fmnist"] = mycnn_FashionMNIST(batch_size=args.batch_size)


    # from utils.load_utils import compare_validations
    # compare_validations(all_acc_dict)