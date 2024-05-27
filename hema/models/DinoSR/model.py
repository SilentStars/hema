import torch

from EMA import EMA
from utils import compute_mask_indices, index_put, GradMultiply 
from nn.conv_feature_extractor import FeatureExtractor

# class DinoSR_config():
#     def __init__(
#         self
#     ):
#         super

class DinoSR(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        





