import torch
from hinge_utils import one_hot_to_hinge_labels


def l01c(f1, f2, target, s,c):
    # c = torch.where(s>= 0.5, c, 0)
    target = one_hot_to_hinge_labels(target)

    f1_d = torch.where(f1 > 0, 1, -1)
    f2_d = torch.where(f2 > 0, 1, -1)
    f1_pen = torch.where(f1_d != target, 1, 0)
    f1_s_pen = torch.where(s <= 0.5, f1_pen, 0)
    f2_pen = torch.where(f2_d != target, 1, 0)
    f2_s_pen = torch.where(s > 0.5, f2_pen + c, 0)
    rd = torch.where(s > 0.5, 1, 0)
    rd_gt = torch.where(f1_d != target, torch.where(f2_d == target, 1, 0), 0)
    defer_acc = torch.where(rd == rd_gt, 1, 0)
    # breakpoint()
    return {
            'l01c loss' : torch.sum(f1_s_pen + f2_s_pen), 
            'f1 penalty' : f1_pen ,
            'f2 penalty' : f2_pen,
            'f1 selected penalty' : f1_s_pen,
            'f2 selected penalty' : f2_s_pen,
            'rate of deferral':torch.sum(rd),
            'gt rate of deferral': torch.sum(rd_gt),
            'deferral accuracy': defer_acc
            }

