import torch

from conv_blocks import DownsampleBlock
from typing import List, Tuple

class FeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        in_dim: int = 8,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.conv_layers = torch.nn.ModuleList()
        
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3
            (dim, kernel_size, stride) = cl
            
            self.conv_layers.append(
                DownsampleBlock(self.in_dim, dim, kernel_size, stride=stride)
            )
            
            self.in_dim = dim
    
    def forward(self, x):
        # shape of x should be (Batch * Channels * Length)
        for conv in self.conv_layers:
            x = conv(x)
        
        return x