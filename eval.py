import torch

def l01c(f1, f2, target, s,c):
    # c = torch.where(s>= 0.5, c, 0)
    f1_pen = torch.where(f1 != target, 1, 0)
    f1_s_pen = torch.where(s <= 0.5, f1_pen, 0)
    f2_pen = torch.where(f2 != target, 1, 0)
    f2_s_pen = torch.where(s >= 0.5, f2_pen + c, 0)
    
    return {
            'l01c loss' : torch.sum(f1_s_pen + f2_s_pen), 
            'f1 penalty' : f1_pen ,
            'f2 penalty' : f2_pen,
            'f1 selected penalty' : f1_s_pen,
            'f2 selected penalty' : f2_s_pen 
            }

