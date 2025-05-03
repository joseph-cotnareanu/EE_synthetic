import torch 
from tqdm import tqdm
from eval_utils import compute_accuracies_and_01c, get_pred, l01c
from storing_plotting import plot_xzy
from training.loss import loss_hinge_joint, sep_hinge, loss_CE_joint



def train_two_stage_experiment(train_loader, test_loader, cost, two_stage_model, training_configs):
    
    epoch = training_configs['epoch']
    batch_size = training_configs['batch_size']
    lr = training_configs['lr']
    loss_type = training_configs['loss_type']
    
    optimizer = torch.optim.Adam(two_stage_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    track_batch_loss = []
    track_epoch_loss = []
    track_t1_acc = []
    track_t2_acc = []
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
    for j in tqdm(range(epoch)):
        running_loss = 0
        debug=False
        
        for i, (x_batch, z_batch, y_batch) in enumerate(train_loader):
            if  i%32 == 0:
                test_acc_t1, test_acc_t2, test_01c = compute_accuracies_and_01c(two_stage_model, test_loader, cost)
                track_t1_acc.append(test_acc_t1)
                track_t2_acc.append(test_acc_t2)
                track_l01c.append(test_01c)
           
            optimizer.zero_grad()
            t1, t2, s, param_dict= two_stage_model(x_batch, z_batch, debug=debug)
            if 'c' in param_dict:

                cs.append(param_dict['c'].detach().numpy().item())
                ds.append(param_dict['d'].detach().numpy().item())
            debug=False
            if loss_type == 'separate':
                loss_f1, loss_f2 = sep_hinge(x_batch, z_batch, y_batch, cost, t1, t2, s)
                loss = loss_f1 + loss_f2
                s = 1- torch.abs(t1)
            elif loss_type == 'hinge_surrogate':
                loss, loss_f1, loss_f2 = loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s)
            f1ls.append(loss_f1.detach().numpy().item()/x_batch.shape[0])
            f2ls.append(loss_f2.detach().numpy().item()/x_batch.shape[0])
            ls.append(loss.detach().numpy().item()/x_batch.shape[0])
            f1 = torch.where(t1 >0, 1, 0)
            f2 = torch.where(t2 >0, 1, 0)
            a = l01c(f1, f2, y_batch[:,1], s, cost)['l01c loss']
            # ls.append(loss.detach().numpy().item()/x_batch.shape[0])
            l01cs.append(a.detach().numpy().item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            track_batch_loss.append(loss.item())
            
        
       
        avg_loss = running_loss/ len(train_loader.dataset)
        track_epoch_loss.append(avg_loss)
        print(f"Epoch {j+1}/{epoch}, Loss: {avg_loss}")
        scheduler.step()
     
    t1_all, t2_all, y_all, x_all, z_all, s_all, gt_s_all, gt_f1_all, gt_f2_all = get_pred(two_stage_model, test_loader,cost)


    print('average defferal to f2:', torch.mean(s_all))
    print('average ground truth defferal to f2:', torch.mean(gt_s_all))
    plot_xzy(x_all, z_all,s_all, prefix=str(cost)+'_s_')
    plot_xzy(x_all, z_all,gt_s_all, prefix=str(cost)+'_gt_s_')
    plot_xzy(x_all, z_all, y_all[:,1], prefix=str(cost)+'_gt_')
    plot_xzy(x_all, z_all, t1_all, prefix=str(cost)+'_t1_')
    plot_xzy(x_all, z_all, t2_all, prefix=str(cost)+'_t2_')

    
    optimal_l01c =l01c(gt_f1_all, gt_f2_all, y_all[:,1], gt_s_all, cost)['l01c loss'].detach().item()
    f1_all = torch.where(t1_all >0, 1, 0)
    f2_all = torch.where(t2_all >0, 1, 0)
    lout = l01c(f1_all, f2_all, y_all[:,1], s_all, cost)
    l01cs_test =lout['l01c loss'].detach().item()
    df_testacc = torch.sum(lout['deferral accuracy'])/len(y_all)
    f1_testpen= lout['f1 selected penalty']
    f2_testpen = lout['f2 selected penalty']
    df_testrate = torch.sum(lout['rate of deferral'])/len(y_all)

    f1_testacc = torch.sum(lout['f1 acc'])/len(y_all)
    f2_testacc = torch.sum(lout['f2 acc'])/len(y_all)
     
    
    training_log_dict['param_cs'] = cs
    training_log_dict['param_ds'] = ds
    training_log_dict['track_t1_acc'] = track_t1_acc
    training_log_dict['track_t2_acc'] = track_t2_acc
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


