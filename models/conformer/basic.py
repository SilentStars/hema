import torch
import torch.nn as nn
import torch.nn.init as init

from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
    
class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape
    
    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)