import torch 
from tqdm import tqdm
from eval_utils import get_pred, l01c_multi
from training.loss import multi_class_loss_hinge_joint, loss_CE_joint_multi


def train_two_stage_experiment(train_loader, test_loader, cost, two_stage_model, training_configs, device='cpu'):
    datatype = 'toy_multi'
    epoch = training_configs['epoch']
    batch_size = training_configs['batch_size']
    lr = training_configs['lr']
    loss_type = training_configs['loss_type']
    n_classes = training_configs['n_classes']
    
    optimizer = torch.optim.Adam(two_stage_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    track_batch_loss = []
    track_epoch_loss = []
    track_t1_acc = []
    track_t2_acc = []
    track_avg_s = []
    track_l01c = []
    training_log_dict = {}
    cs = []
    ds = []
    f1ls = []
    f2ls = []
    ls = []
    l01cs = []
    l01cs_test = []
    df_testacc = []
    df_testrate = []
    
    t1_all, t2_all, y_all, x_all, z_all, s_all, gt_s_all, gt_f1_all, gt_f2_all = get_pred(two_stage_model, test_loader, data=datatype,cost=cost, device=device)
    
    s_all_init = s_all.detach().clone()

    print('PRE TRAINING -----------')
    print('average defferal to f2:', torch.mean(s_all))
    print('average ground truth defferal to f2:', torch.mean(gt_s_all))
    print('-----------')
    
    for j in range(epoch):
        running_loss = 0
        debug=False
        
        
        for i, (x_batch, z_batch, y_batch) in tqdm(enumerate(train_loader), total=len(train_loader)):
    
            optimizer.zero_grad()
            # move data to device
            x_batch = x_batch.to(device)
            z_batch = z_batch.to(device)
            y_batch = y_batch.to(device)
            
            
            t1, t2, s, param_dict= two_stage_model(x_batch, z_batch, debug=debug)
            
            
            if 'c' in param_dict:
                cs.append(param_dict['c'].detach().numpy().item())
                ds.append(param_dict['d'].detach().numpy().item())
            debug=False
            if loss_type == 'separate':
                _, loss_f1, loss_f2 = multi_class_loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s, nclasses=n_classes)
                loss = loss_f1 + loss_f2
                s = 1- torch.nn.Softmax(dim=-1)(t1).max(dim=-1).values.reshape(-1, 1)
            elif loss_type == 'hinge_surrogate':
                loss, loss_f1, loss_f2 = multi_class_loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s, nclasses=n_classes)
            elif loss_type == 'CE_multi':
                loss, loss_f1, loss_f2 = loss_CE_joint_multi(x_batch, z_batch, y_batch, cost, t1, t2, s)
            f1ls.append(loss_f1.detach().cpu().numpy().item()/x_batch.cpu().shape[0])
            f2ls.append(loss_f2.detach().cpu().numpy().item()/x_batch.cpu().shape[0])
            ls.append(loss.detach().cpu().numpy().item()/x_batch.cpu().shape[0])
            
            f1 = torch.max(t1, dim=-1).indices
            f2 = torch.max(t2, dim=-1).indices
            lout = l01c_multi(f1, f2, y_batch, s, cost)
            a = lout['l01c loss']
            
            l01cs.append(a.detach().cpu().numpy().item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            track_batch_loss.append(loss.item())
        
        #test data at end of epoch
        t1_all, t2_all, y_all, x_all, z_all, s_all, gt_f1_all, gt_f2_all = get_pred(two_stage_model, test_loader, data=datatype, device=device)
                     
        f1_all = torch.max(t1_all, dim=-1).indices
        f2_all = torch.max(t2_all , dim=-1).indices
        test_01c = l01c_multi(f1_all, f2_all, y_all, s_all, cost)['l01c loss']
        
        #convert one hot labels to hinge labels
        y_hinge = torch.max(y_all, dim=-1).indices
        test_acc_t1 = torch.mean(torch.where(f1_all == y_hinge, 1, 0).float())
        test_acc_t2 = torch.mean(torch.where(f2_all == y_hinge, 1, 0).float())
        
        track_t1_acc.append(test_acc_t1)
        track_t2_acc.append(test_acc_t2)
        track_l01c.append(test_01c)
        track_avg_s.append(torch.where(s_all > 0.5, 1, 0).float().mean().cpu().item())
            
            
            
        
       
        avg_loss = running_loss/ len(train_loader.dataset)
        track_epoch_loss.append(avg_loss)
        print(" ")
        print(f"Epoch {j+1}/{epoch}, Loss: {avg_loss}")
        scheduler.step()
        
        
        
    
    
    t1_all, t2_all, y_all, x_all, z_all, s_all, gt_s_all, gt_f1_all, gt_f2_all = get_pred(two_stage_model, test_loader, data=datatype, cost=cost, device=device)
    
    print('POST TRAINING -----------')
    print('average defferal to f2:', torch.mean(s_all))
    print('average ground truth defferal to f2:', torch.mean(gt_s_all))
    print('-----------')
    print('\n\n')

    
    optimal_l01c =l01c_multi(gt_f1_all, gt_f2_all, y_all, gt_s_all, cost)['l01c loss'].detach().item()
    f1_all = torch.max(t1_all, dim=-1).indices
    f2_all = torch.max(t2_all , dim=-1).indices
    lout = l01c_multi(f1_all, f2_all, y_all, s_all, cost)
    l01cs_test =lout['l01c loss'].detach().item()
        
    df_testacc = torch.sum(lout['deferral accuracy'])/len(y_all)
    
    df_testrate = torch.sum(lout['rate of deferral'])/len(y_all)

    f1_testacc = torch.sum(lout['f1 acc'])/len(y_all)
    f2_testacc = torch.sum(lout['f2 acc'])/len(y_all)
    
    training_log_dict['xzy_x'] = x_all
    training_log_dict['xzy_z'] = z_all
    training_log_dict['xzy_s'] = s_all
    training_log_dict['xzy_gt_s'] = gt_s_all
    training_log_dict['xzy_y'] = y_all
    training_log_dict['xzy_t1'] = t1_all
    training_log_dict['xzy_t2'] = t2_all
    
    training_log_dict['xzy_init_s'] = s_all_init
     
    training_log_dict['param_cs'] = cs
    training_log_dict['param_ds'] = ds
    training_log_dict['track_t1_acc'] = track_t1_acc
    training_log_dict['track_t2_acc'] = track_t2_acc
    training_log_dict['track_avg_s'] = track_avg_s
    training_log_dict['track_01c'] = track_l01c
    training_log_dict['ls'] = ls
    training_log_dict['f1ls'] = f1ls
    training_log_dict['f2ls'] = f2ls
    training_log_dict['l01cs'] = l01cs
    training_log_dict['test_avg_l01c'] = l01cs_test
    training_log_dict['optimal_l01c'] = optimal_l01c
    training_log_dict['track_epoch_loss'] = track_epoch_loss
    
    training_log_dict['df_testacc'] = df_testacc
    training_log_dict['df_testrate'] = df_testrate
    training_log_dict['f1 acc'] = f1_testacc
    training_log_dict['f2 acc'] = f2_testacc

    return two_stage_model, training_log_dict, 


