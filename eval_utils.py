
import torch


def l01c(f1, f2, target, s,c):
    """
    f1 shoul dbe 0-1 labels
    f2 should be 0-1 labels
    target should be 0-1 labels
    s should be between 0 and 1
    """
    # ensure that they gave the same shape
    f1 = f1.reshape(target.shape).to(torch.float)
    f2 = f2.reshape(target.shape).to(torch.float)
    s = s.reshape(target.shape).to(torch.float)
    target = target.to(torch.float)
    f1_pen = torch.where(f1 != target, 1.0, 0.0)
    f1_s_pen = torch.where(s <= 0.5, f1_pen, 0.0)
    f2_pen = torch.where(f2 != target, 1.0, 0.0)
    f2_s_pen = torch.where(s > 0.5, f2_pen + c, 0.0)
    rd = torch.where(s > 0.5, 1.0, 0.0)
    rd_gt = torch.where(f1 != target, torch.where(f2 == target, 1, 0), 0.0)
    defer_acc = torch.where(rd == rd_gt, 1.0, 0.0)
    f1_acc = 1-f1_pen
    f2_acc = 1 - (f2_pen/(1+c))
    # breakpoint()
    return {
            'l01c loss' : torch.mean(f1_s_pen + f2_s_pen), 
            'f1 penalty' : f1_pen ,
            'f2 penalty' : f2_pen,
            'f1 acc': f1_acc,
            'f2 acc': f2_acc,
            'f1 selected penalty' : f1_s_pen,
            'f2 selected penalty' : f2_s_pen,
            'rate of deferral':torch.sum(rd),
            'gt rate of deferral': torch.sum(rd_gt),
            'deferral accuracy': defer_acc

            }

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

def compute_gt_f1_f2(test_py_xz, test_py_x):
    f1_star_y = (test_py_x >= 0.5).float()
    f2_star_y = (test_py_xz >= 0.5).float()
    return f1_star_y, f2_star_y

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
    t1_list, t2_list, x_list, z_list, y_stack, s_stack, gt_s, gt_f1, gt_f2 = [], [], [], [], [], [], [], [], []
    
    with torch.no_grad():
        for i, (x_batch, z_batch, y_batch, E_max_py_xz_batch, max_y_batch, test_py_xz, test_py_x) in enumerate(test_loader):
            
            if cost is not None:
                
                gt_s_batch = compute_gt_s(E_max_py_xz_batch, max_y_batch, cost)
                gt_s.append(gt_s_batch)
            
            gt_f1_batch, gt_f2_batch = compute_gt_f1_f2(test_py_xz, test_py_x)
            gt_f1.append(gt_f1_batch)
            gt_f2.append(gt_f2_batch)
            
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
        gt_f1_all = torch.cat(gt_f1, dim=0)
        gt_f2_all = torch.cat(gt_f2, dim=0)
        if cost is not None:
            gt_s_all = torch.cat(gt_s, dim=0)
            return t1_all, t2_all, y_all, x_all, z_all, s_all, gt_s_all, gt_f1_all, gt_f2_all
        return t1_all, t2_all, y_all, x_all, z_all, s_all, gt_f1_all, gt_f2_all
        

def compute_accuracies_and_01c(two_stage_model, test_loader, c ):
    t1_all, t2_all, y_all, x_all, z_all, s_all, gt_f1_all, gt_f2_all = get_pred(two_stage_model, test_loader)
    f1_all = torch.where(t1_all >0, 1, 0)
    f2_all = torch.where(t2_all >0, 1, 0)
    test_01c = l01c(f1_all,f2_all,y_all[:,1], s_all, c)['l01c loss']
    y_hinge = one_hot_to_hinge_labels(y_all)
    return accuracy_hinge_model(t1_all, y_hinge), accuracy_hinge_model(t2_all, y_hinge), test_01c.item()
