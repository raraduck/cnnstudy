import torch
from torch import nn, optim

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
                                        nn.BatchNorm2d(out_channels, eps=0.001),
                                        nn.ReLU())
    def forward(self, x):
        x = self.conv_block(x)
        return x

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super().__init__()

        self.branch3x3dbl = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1), # dbl = double
                                          BasicConv2d(64, 96, kernel_size=3, padding=1),
                                          BasicConv2d(96, 96, kernel_size=3, padding=1))

        self.branch3x3 = nn.Sequential(BasicConv2d(in_channels, 48, kernel_size=1),
                                       BasicConv2d(48, 64, kernel_size=3, padding=1))

        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         BasicConv2d(in_channels, pool_features, kernel_size=1))

        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        x = [self.branch3x3dbl(x), self.branch3x3(x), self.branch_pool(x), self.branch1x1(x)]
        return torch.cat(x,1)

class ReductionA(nn.Module): # Bottleneck 피하면서 grid-size 줄이기
    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3dbl = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1),
                                          BasicConv2d(64, 96, kernel_size=3, padding=1),
                                          BasicConv2d(96, 96, kernel_size=3, stride=2))

        self.branch3x3 = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1),
                                       BasicConv2d(64, 384, kernel_size=3, stride=2))


        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = [self.branch3x3dbl(x), self.branch3x3(x), self.branch_pool(x)]
        return torch.cat(x,1)

class InceptionB(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super().__init__()

        c7 = channels_7x7
        self.branch7x7dbl = nn.Sequential(BasicConv2d(in_channels, c7, kernel_size=1),
                                          BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)), # 7x1, 1x7 순으로 되어있던 것을 논문이랑 같게 순서 바꿈
                                          BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
                                          BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
                                          BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0)))

        self.branch7x7 = nn.Sequential(BasicConv2d(in_channels, c7, kernel_size=1),
                                       BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
                                       BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0)))

        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         BasicConv2d(in_channels, 192, kernel_size=1))

        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        x = [self.branch7x7dbl(x), self.branch7x7(x), self.branch_pool(x), self.branch1x1(x)]
        return torch.cat(x,1)

class ReductionB(nn.Module): # Bottleneck 피하면서 grid-size 줄이기
    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3dbl = nn.Sequential(BasicConv2d(in_channels, 192, kernel_size=1),
                                          BasicConv2d(192, 192, kernel_size=3, padding=1),
                                          BasicConv2d(192, 192, kernel_size=3, stride=2))

        self.branch3x3 = nn.Sequential(BasicConv2d(in_channels, 192, kernel_size=1),
                                       BasicConv2d(192, 320, kernel_size=3, stride=2))

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = [self.branch3x3dbl(x), self.branch3x3(x), self.branch_pool(x)]
        return torch.cat(x,1)

class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3dbl = nn.Sequential(BasicConv2d(in_channels, 448, kernel_size=1),
                                          BasicConv2d(448, 384, kernel_size=3, padding=1))
        self.branch3x3dbla = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dblb = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         BasicConv2d(in_channels, 192, kernel_size=1))

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

    def forward(self, x):
        branch3x3dbl = self.branch3x3dbl(x)
        branch3x3dbl = [self.branch3x3dbla(branch3x3dbl),
                        self.branch3x3dblb(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch3x3 = self.branch3x3(x)
        branch3x3 = [self.branch3x3a(branch3x3),
                     self.branch3x3b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)

        branch_pool = self.branch_pool(x)

        branch1x1 = self.branch1x1(x)

        outputs = [branch3x3dbl, branch3x3, branch_pool, branch1x1]
        return torch.cat(outputs,1)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.avgpool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv2a = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv2b = BasicConv2d(128, 1024, kernel_size=1) # FC 라고 써있긴 한데,
        # GAP를 해야 image size가 달라졌을 때 error가 안날거라 GAP 가 어딘가 쓰이는 게 좋을 거 같은데
        # GAP 를 먼저 하자니 128x1x1 로 너무 확 줄어서 1024로 늘린 다음 GAP 하는 것으로 1024x1x1을 만들었다
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = self.avgpool1(x)
        # N x 768 x 5 x 5
        x = self.conv2a(x)
        # N x 128 x 5 x 5
        x = self.conv2b(x)
        # N x 1024 x 5 x 5
        x = self.avgpool2(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.fc(x)
        # N x 1000
        return x

class Inception_V3(nn.Module):
    def __init__(self, num_classes = 1000, use_aux = True, init_weights = None, drop_p = 0.5):
        super().__init__()

        self.use_aux = use_aux

        self.conv1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv1b = BasicConv2d(32, 32, kernel_size=3)
        self.conv1c = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2a = BasicConv2d(64, 80, kernel_size=3)
        self.conv2b = BasicConv2d(80, 192, kernel_size=3, stride=2)
        self.conv2c = BasicConv2d(192, 288, kernel_size=3, padding=1)

        self.inception3a = InceptionA(288, pool_features=64)
        self.inception3b = InceptionA(288, pool_features=64)
        self.inception3c = ReductionA(288)

        self.inception4a = InceptionB(768, channels_7x7=128)
        self.inception4b = InceptionB(768, channels_7x7=160)
        self.inception4c = InceptionB(768, channels_7x7=160)
        self.inception4d = InceptionB(768, channels_7x7=192)
        if use_aux:
            self.aux = InceptionAux(768, num_classes)
        else:
            self.aux = None
        self.inception4e = ReductionB(768)

        self.inception5a = InceptionC(1280)
        self.inception5b = InceptionC(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.conv1a(x)
        # N x 32 x 149 x 149
        x = self.conv1b(x)
        # N x 32 x 147 x 147
        x = self.conv1c(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.conv2a(x)
        # N x 80 x 71 x 71
        x = self.conv2b(x)
        # N x 192 x 35 x 35
        x = self.conv2c(x)
        # N x 288 x 35 x 35
        x = self.inception3a(x)
        # N x 288 x 35 x 35
        x = self.inception3b(x)
        # N x 288 x 35 x 35
        x = self.inception3c(x)
        # N x 768 x 17 x 17
        x = self.inception4a(x)
        # N x 768 x 17 x 17
        x = self.inception4b(x)
        # N x 768 x 17 x 17
        x = self.inception4c(x)
        # N x 768 x 17 x 17
        x = self.inception4d(x)
        # N x 768 x 17 x 17

        if self.aux is not None and self.training:
            aux = self.aux(x)
        else:
            aux = None  # 뭐라도 넣어놔야 not defined error 안 뜸

        x = self.inception4e(x)
        # N x 1280 x 8 x 8
        x = self.inception5a(x)
        # N x 2048 x 8 x 8
        x = self.inception5b(x)
        # N x 2048 x 8 x 8
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux