import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, LeakyReLU, Conv1d, GELU

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        
class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

class DownsampleBlock(nn.Module):
    def __init__(self, num_ins, num_outs, kernel_size=5, stride=2):
        super().__init__()
        
        self.conv = nn.Sequential(
            Conv1d(num_ins, num_outs, kernel_size, 1, padding=int(kernel_size/2), bias=False),
            nn.Sequential(
                TransposeLast(),
                nn.LayerNorm(num_outs),
                TransposeLast(),
            ),
            GELU(),
            Conv1d(num_outs, num_outs, kernel_size, 1, padding=int(kernel_size/2), bias=False),
            nn.Sequential(
                TransposeLast(),
                nn.LayerNorm(num_outs),
                TransposeLast(),
            ),
            GELU(),
            Conv1d(num_outs, num_outs, kernel_size, stride, padding=int(kernel_size/2), bias=False),
            nn.Sequential(
                TransposeLast(),
                nn.LayerNorm(num_outs),
                TransposeLast(),
            ),
        )
        self.ac = GELU()

        self.conv.apply(weights_init)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.ac(x)
        
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, num_ins, num_outs, kernel_size=5, stride=2, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        
        self.conv = nn.Sequential(
            Conv1d(num_ins, num_outs, kernel_size, 1, padding=int(kernel_size/2), bias=False),
            BatchNorm1d(num_outs),
            Conv1d(num_outs, num_outs, kernel_size, 1, padding=int(kernel_size/2), bias=False),
            BatchNorm1d(num_outs),
        )
        self.out = Conv1d(num_outs, num_outs, kernel_size, 1, padding=int(kernel_size/2), bias=False)
        self.bn = BatchNorm1d(num_outs)
        self.ac = LeakyReLU(negative_slope=1e-2, inplace=True)

        self.conv.apply(weights_init)
        
    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='linear')
        x = self.out(x)
        if not self.last_layer:
            x = self.ac(self.bn(x))
        else:
            x = x
        
        return x
    