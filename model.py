import torch
import torch.nn as nn
from layers import MaskedConv2d

class PixelCNN(nn.Module):
  def __init__(self, n_channel=3, h=128, discrete_channel=256):
    super().__init__()
    self.discrete_channel = discrete_channel

    self.MaskAConv = MaskAConvBlock(n_channel, 2 * h, k_size=7, stride=1, pad=3)
    MaskBConv = []
    for i in range(1):
      MaskBConv.append(MaskBConvBlock(h, k_size=3, stride=1, pad=1))
    self.MaskBConv = nn.Sequential(*MaskBConv)

    self.out = nn.Sequential(
        nn.Conv2d(2 * h, 1024, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.Conv2d(1024, n_channel * discrete_channel, kernel_size=1, stride=1, padding=0))

  def forward(self, x):
    batch_size, c_in, height, width = x.size()
    # [BATCH_SIZE, 2*h, 32, 32]
    x = self.MaskAConv(x)
    # [BATCH_SIZE, 2*h, 32, 32]
    x = self.MaskBConv(x)
    # [BATCH_SIZE, 3*256, 32, 32]
    x = self.out(x)
    # [BATCH_SIZE, 3, 256, 32, 32]
    x = x.view(batch_size, c_in, self.discrete_channel, height, width)
    # [BATCH_SIZE, 3, 32, 32, 256]
    x = x.permute(0, 1, 3, 4, 2)
    return x
