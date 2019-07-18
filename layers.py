import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_ch, out_ch, kernel, stride, padding):
        # variable names to be passed from subclass to parent class
        super().__init__(in_ch, out_ch, kernel, stride, padding, bias=False)

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
