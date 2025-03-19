import torch
from torch.nn.functional import relu
from sklearn.metrics import hinge_loss
from hinge_utils import one_hot_to_hinge_labels
from torchmetrics import HingeLoss
hinge=HingeLoss(task='binary')
def binary_hinge_loss(t,y):
    """
    hinge loss: max(0,1-t*y)
    """
    # return torch.ones_like(y)
    # return y
    return relu(1-torch.mul(t,y))
    # hinge=HingeLoss(task='binary')
    # return hinge(t, torch.where(y==-1, 0, y))



def loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s):
    # y_batch to 1 -1 labels
    y_hinge = one_hot_to_hinge_labels(y_batch)
    hinge_f1 = binary_hinge_loss(t1,y_hinge) 
    hinge_f2 = binary_hinge_loss(t2,y_hinge) 
    surrogate_loss = (1-s) * hinge_f1 + s * (hinge_f2 + 2*cost)
    # return sum(surrogate_loss)
    if len(surrogate_loss.shape) == 0: return surrogate_loss, hinge_f1, hinge_f2
    else: return sum(surrogate_loss), torch.sum(hinge_f1), torch.sum(hinge_f2)
