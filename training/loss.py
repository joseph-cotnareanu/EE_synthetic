import torch
from torch.nn.functional import relu

from hinge_utils import one_hot_to_hinge_labels

def binary_hinge_loss(t,y):
    """
    hinge loss: max(0,1-t*y)
    """
    return relu(1-torch.mul(t,y))


def loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s):
    # y_batch to 1 -1 labels
    y_hinge = one_hot_to_hinge_labels(y_batch)
    hinge_f1 = binary_hinge_loss(t1,y_hinge)
    hinge_f2 = binary_hinge_loss(t2,y_hinge)
    surrogate_loss = (1-s) * hinge_f1 + s * (hinge_f2 + 2*cost)
    return sum(surrogate_loss)
