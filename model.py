import torch
import torch.nn as nn
from layers import MaskedConv2d

class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(MaskedConv2d('A', 1, fm, 7, 1, 3), nn.BatchNorm2d(fm), nn.ReLU(True),
                            MaskedConv2d('B', fm, fm, 7, 1, 3), nn.BatchNorm2d(fm), nn.ReLU(True),
                            MaskedConv2d('B', fm, fm, 7, 1, 3), nn.BatchNorm2d(fm), nn.ReLU(True),
                            MaskedConv2d('B', fm, fm, 7, 1, 3), nn.BatchNorm2d(fm), nn.ReLU(True),
                            MaskedConv2d('B', fm, fm, 7, 1, 3), nn.BatchNorm2d(fm), nn.ReLU(True),
                            MaskedConv2d('B', fm, fm, 7, 1, 3), nn.BatchNorm2d(fm), nn.ReLU(True),
                            MaskedConv2d('B', fm, fm, 7, 1, 3), nn.BatchNorm2d(fm), nn.ReLU(True),
                            MaskedConv2d('B', fm, fm, 7, 1, 3), nn.BatchNorm2d(fm), nn.ReLU(True),
                            nn.Conv2d(fm, 1, 1))
    def forward(self, x):
        out = self.net(x)
        return out
