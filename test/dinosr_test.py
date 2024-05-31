from hema.models.DinoSR.model import DinoSR, config
import torch

if __name__ == '__main__':
    cfg = config()
    model = DinoSR(cfg)
    
    # input shape B * C * T
    fake_emg = torch.rand((12, 8, 400))
    res = model.forward(fake_emg)
    
    print(res)
    