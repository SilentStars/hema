import hema.models.DinoSR.model as dino

import torch

if __name__ == '__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = dino.config(encoder_layers=6, average_top_k_layers=4)
    model = dino.DinoSR(cfg)
    
    for i in range(0, 10):
        # input shape (B T C)
        rand_ts = torch.rand((16, 8, 1600)).to('cuda')
        model.to('cuda')
        model.set_num_updates(i)
        # model.to('cuda')
        res = model(rand_ts)
        
        print(res)
