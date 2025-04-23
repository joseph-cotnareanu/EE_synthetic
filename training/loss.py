import torch
from torch.nn.functional import relu
from torch.nn import BCELoss
from sklearn.metrics import hinge_loss
from eval_utils import one_hot_to_hinge_labels
from torchmetrics import HingeLoss
hinge=HingeLoss(task='binary')
multi_class_hinge_loss = torch.nn.MultiMarginLoss(p=1, margin=1, weight=None, size_average=None, reduce=None)

def binary_hinge_loss(t,y):
    """
    hinge loss: max(0,1-t*y)
    """
    # return torch.ones_like(y)
    # return y
    return relu(1-torch.mul(t,y))
    # hinge=HingeLoss(task='binary')
    # return hinge(t, torch.where(y==-1, 0, y))

def binary_CE_loss(t,y):
    """
    cross entropy loss: t/2 + 0.5 
    """
    p_1 = t/2 + 0.5 
    p = torch.concat((1-p_1, p_1), dim=1)
    
    return BCELoss()(p, y.to(torch.float32))

def loss_CE_joint(x_batch, z_batch, y_batch, cost, t1, t2, s):
    # y_batch to 1 -1 labels
  
    ce_f1 = binary_CE_loss(t1,y_batch) 
    ce_f2 = binary_CE_loss(t2,y_batch) 
    surrogate_loss = (1-s) * ce_f1 + s * (ce_f2 + cost)
    # return sum(surrogate_loss)
    if len(surrogate_loss.shape) == 0: return surrogate_loss, ce_f1, ce_f2
    else: return sum(surrogate_loss), torch.sum(ce_f1), torch.sum(ce_f2)


def loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s):
    # y_batch to 1 -1 labels
    y_hinge = one_hot_to_hinge_labels(y_batch)
    hinge_f1 = binary_hinge_loss(t1,y_hinge) 
    hinge_f2 = binary_hinge_loss(t2,y_hinge) 
    surrogate_loss = (1-s) * hinge_f1 + s * (hinge_f2 + 2*cost)
    # return sum(surrogate_loss)
    if len(surrogate_loss.shape) == 0: return surrogate_loss, hinge_f1, hinge_f2
    else: return sum(surrogate_loss), torch.sum(hinge_f1), torch.sum(hinge_f2)

def sep_hinge(x_batch, z_batch, y_batch, cost, t1, t2, s):

    y_hinge = one_hot_to_hinge_labels(y_batch)  

    f1l = binary_hinge_loss(t1, y_hinge)
    f2l = binary_hinge_loss(t2, y_hinge)

    return torch.sum(f1l), torch.sum(f2l)

def mc_hinge(t,y):
    return multi_class_hinge_loss(t, y)
    
def multi_class_loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s):
    y_batch = torch.max(y_batch, dim=-1).indices
    hinge_f1 = mc_hinge(t1,y_batch) 
    hinge_f2 = mc_hinge(t2,y_batch) 
    # breakpoint()
    surrogate_loss = (1-s) * hinge_f1 + s * (hinge_f2 + cost)
    # surrogate_loss = (1-s) * hinge_f1 + s * (hinge_f2)

    # return sum(surrogate_loss)
    if len(surrogate_loss.shape) == 0: return surrogate_loss, hinge_f1, hinge_f2
    else: return sum(surrogate_loss), torch.sum(hinge_f1), torch.sum(hinge_f2)