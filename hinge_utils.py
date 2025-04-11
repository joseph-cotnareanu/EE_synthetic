
import torch
def one_hot_to_hinge_labels(y_one_hot):
    """
    y = [[0,1],[1,0],[0,1], ....]
    to
    y = [1,-1,1,...]
    """
    y_hinge = y_one_hot.clone()
    y_hinge[:,0] = y_hinge[:,0]*-1
    y_hinge = torch.sum(y_hinge, axis=1)[:,None]
    return y_hinge

def compute_gt_s(E_max_py_xz, max_y_x, c):
    """
    E_max_py_xz = E[max(p(y=1 | x, z), p(y=0 | x, z)]
    max_y_x = max(p(y=1 | x), p(y=0 | x))
    
    # return 0 (use f1) if max_y_x > E_max_py_xz - c 
    # return 1 (use f2) if max_y_x < E_max_py_xz - c
    """
    gt_s = (torch.tensor(max_y_x) < (torch.tensor(E_max_py_xz) - c)).float()
    return gt_s

def accuracy_hinge_model(y_pred, y_true):
    """
    y_pred = some real values [2,-4,3.4,..]
    y_true = binary classes [1,1,-1,-1,...]
    """
    y_pred_labels = torch.sign(y_pred)  
    y_pred_labels[y_pred_labels == 0] = 1  # Convert 0 predictions to 1
    correct = (y_pred_labels == y_true).float()
    return correct.mean().item()

def get_pred(two_stage_model, test_loader, cost=None):
    t1_list, t2_list, x_list, z_list, y_stack, s_stack, gt_s = [], [], [], [], [], [], []
    
    with torch.no_grad():
        for i, (x_batch, z_batch, y_batch, E_max_py_xz_batch, max_y_batch, _) in enumerate(test_loader):
            
            if cost is not None:
                
                gt_s_batch = compute_gt_s(E_max_py_xz_batch, max_y_batch, cost)
                gt_s.append(gt_s_batch)
            t1, t2, s, _ = two_stage_model(x_batch, z_batch, debug=False)
            
            x_list.append(x_batch)
            z_list.append(z_batch)
            s_stack.append(s)
            t1_list.append(t1)
            t2_list.append(t2)
            y_stack.append(y_batch)
            
        x_all = torch.cat(x_list, dim=0)
        s_all = torch.cat(s_stack, dim=0)
        z_all = torch.cat(z_list, dim=0)
        t1_all = torch.cat(t1_list, dim=0)
        t2_all = torch.cat(t2_list, dim=0)
        y_all = torch.cat(y_stack, dim=0)
        if cost is not None:
            gt_s_all = torch.cat(gt_s, dim=0)
            return t1_all, t2_all, y_all, x_all, z_all, s_all, gt_s_all
        return t1_all, t2_all, y_all, x_all, z_all, s_all
        

def compute_accuracies(two_stage_model, test_loader ):
    t1_all, t2_all, y_all, x_all, z_all, s_all = get_pred(two_stage_model, test_loader)
    y_hinge = one_hot_to_hinge_labels(y_all)
    return accuracy_hinge_model(t1_all, y_hinge), accuracy_hinge_model(t2_all, y_hinge)
