import hema.models.DinoSR.model as dino

import torch

if __name__ == '__main__':
    cfg = dino.config()
    model = dino.DinoSR(cfg)
    
    # input shape (B T C)
    rand_ts = torch.rand((16, 8, 1600))
    res = model(rand_ts)
    
    print(res)
