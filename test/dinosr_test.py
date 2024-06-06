import hema.models.DinoSR.model as dino

import torch

if __name__ == '__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = dino.config()
    model = dino.DinoSR(cfg)
    
    # input shape (B T C)
    rand_ts = torch.rand((16, 8, 1600)).to('cuda')
    model.to('cuda')
    model.set_num_updates(1)
    # model.to('cuda')
    res = model(rand_ts)
    
    print(res)
