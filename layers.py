import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_ch, out_ch, kernel, stride, padding, bias=False):
        # variable names to be passed from subclass to parent class
        super(MaskedConv2d, self).__init__(in_ch, out_ch, kernel, stride, padding)

        self.mask_type = mask_type

        out_ch, in_ch, height, width = self.weight.size()
        mask = torch.ones(out_ch, in_ch, height, width)
        self.register_buffer('mask', mask)

        if mask_type == 'A':
            mask[:, :, height // 2, width // 2 + 1:] = 0
            mask[:, :, height // 2 + 1:] = 0
        else:
            mask[:, :, height // 2, width // 2:] = 0
            mask[:, :, height // 2:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        # calling parent class forward
        return super().forward(x)


class MaskBConvBlock(nn.Module):
  def __init__(self, h=64, k_size=3, stride=1, pad=1):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(2 * h, h, 1), # 1x1 conv
        nn.BatchNorm2d(h),
        nn.ReLU(),
        MaskedConv2d('B', h, h, k_size, stride, pad), # 3x3 conv
        nn.BatchNorm2d(h),
        nn.ReLU(),
        nn.Conv2d(h, 2 * h, 1), # 1x1 conv
        nn.BatchNorm2d(2 * h),
        nn.ReLU())

  def forward(self, x):
    # Residual Connections
    out = self.net(x) + x
    return out


def MaskAConvBlock(c_in=3, c_out=256, k_size=7, stride=1, pad=3):
  return nn.Sequential(
      MaskedConv2d('A', c_in, c_out, k_size, stride, pad),
      nn.BatchNorm2d(c_out),
      nn.ReLU())
