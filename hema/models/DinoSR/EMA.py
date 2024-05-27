import copy
from dataclasses import dataclass, field
import logging

import torch

class EMA:
    """Exponential Moving Average"""

    def __init__(self, model: torch.nn.Module, decay: torch.float = 1, device=None):
        self.decay = decay
        self.model = copy.deepcopy(model)
        
        for params in self.model.parameters():
            params.requires_grad_ = False
            
        if device is not None:
            logging.info(f"Copying EMA model to device {device}")
            self.model = self.model.to(device=device)
            
    def restore(self, state_dict):
        self.model.load_state_dict(state_dict, strict=False)
    
    def set_dacay(self, decay: torch.float):
        self.decay = decay
        
    def get_decay(self):
        return self.decay
    
    def _step_internal(self, new_model: torch.nn.Module):
        decay = self.decay
        
        ema_state_dict = {}
        ema_params = (self.model.state_dict)
        
        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = (
                    param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                )
                ema_params[key] = ema_param
            
            if param.shape != ema_param.shape:
                raise ValueError(
                    "incompatible tensor shapes between model param and ema param"
                    + "{} vs. {}".format(param.shape, ema_param.shape)
                )
                
            if "version" in key:
                # Do not decay a model.version pytorch param
                continue
            
            ema_param.mul_(decay)
            ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - decay)
            
        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param
        
        self.restore(ema_state_dict)
    
    @torch.no_grad()
    def step(self, new_model):
        self._step_internal(new_model)
    
    def reverse(self, model: torch.nn.Module):
        d = self.model.state_dict()
        if "_ema" in d:
            del d["_ema"]
        
        model.load_state_dict(d, strict=False)
        return model