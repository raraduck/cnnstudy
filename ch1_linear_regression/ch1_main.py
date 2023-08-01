from models import *
from data.data_gen import load_1d_data
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim


def main(gamma, epoc):
    gamma = gamma
    num_epoc = epoc
    X, Y = load_1d_data()
    model = MyLinear.MyLinear(1, 1)

    losses = []
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=gamma)
    print(list(model.parameters()))
    for i in range(num_epoc):
        optimizer.zero_grad()
        output = model(X)
        # print(output.shape, Y.shape)
        loss = loss_func(output, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(list(model.parameters()))
    plt.plot(losses)
    plt.show()


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hyper-parameters')
    parser.add_argument('--gamma', metavar='val',
                        help='learning rate', default=0.001)
    parser.add_argument('--epoc', metavar='N',
                        help='num of epoc', default=100)
    args = parser.parse_args()
    main(gamma=args.gamma, epoc=args.epoc)
