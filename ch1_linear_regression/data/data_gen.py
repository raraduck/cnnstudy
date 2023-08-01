import torch


def load_1d_data():
    N = 1000
    X = 10 * torch.randn(N, 1)
    # print(X)
    Noise = 2 * torch.randn(N, 1)
    # idx = np.arange(0, N)
    # plt.scatter(idx, X, s=2)
    # plt.scatter(idx, Noise, s=2)
    # plt.axis([0, N, -50, 50])
    # plt.show()
    Y = 2 * (X + Noise) + 3
    # print(Y.shape)
    return X, Y