import torch 
from tqdm import tqdm
from hinge_utils import compute_accuracies, get_pred
from storing_plotting import plot_xzy
from training.loss import loss_hinge_joint, sep_hinge
from eval import l01c



def train_two_stage_experiment(train_loader, test_loader, cost, two_stage_model, training_configs):
    
    epoch = training_configs['epoch']
    batch_size = training_configs['batch_size']
    lr = training_configs['lr']
    
    optimizer = torch.optim.Adam(two_stage_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    track_batch_loss = []
    track_epoch_loss = []
    track_t1_acc = []
    track_t2_acc = []
    training_log_dict = {}
    cs = []
    ds = []
    f1ls = []
    f2ls = []
    ls = []
    l01cs = []
    last_batch=None
    for i in range(epoch):
        running_loss = 0
        debug=False
        

        
        for i, (x_batch, z_batch, y_batch) in enumerate(tqdm(train_loader)):
            
            
            if  i%32 == 0:
                test_acc_t1, test_acc_t2 = compute_accuracies(two_stage_model, test_loader)
                track_t1_acc.append(test_acc_t1)
                track_t2_acc.append(test_acc_t2)
            
           
            optimizer.zero_grad()
            t1, t2, s, param_dict= two_stage_model(x_batch, z_batch, debug=debug)
            if 'c' in param_dict:

                cs.append(param_dict['c'].detach().numpy().item())
                ds.append(param_dict['d'].detach().numpy().item())
            debug=False
            #s = 1
            loss, loss_f1, loss_f2 = loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s)
            f1ls.append(loss_f1.detach().numpy().item()/x_batch.shape[0])
            f2ls.append(loss_f2.detach().numpy().item()/x_batch.shape[0])
            ls.append(loss.detach().numpy().item()/x_batch.shape[0])
            a = l01c(t1, t2, y_batch, s, cost)['l01c loss']
            # ls.append(loss.detach().numpy().item()/x_batch.shape[0])
            l01cs.append(a.detach().numpy().item()/x_batch.shape[0])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            track_batch_loss.append(loss.item())
            

        last_batch=x_batch.clone()
        avg_loss = running_loss/ len(train_loader.dataset)
        track_epoch_loss.append(avg_loss)
        print(f"Epoch {i+1}/{epoch}, Loss: {avg_loss}")
        scheduler.step()
        # t1_all, t2_all, y_all, x_all, z_all, s_all, gt_s_all = get_pred(two_stage_model, x_test ,z_test,y_test,E_max_py_xz, max_y_x, py_xz,cost, batch_size, test_n)
        
        # plot_xzy(x_all, z_all,s_all, prefix='s_')
        # plot_xzy(x_all, z_all, y_all[:,1], prefix='gt_')
        # plot_xzy(x_all, z_all, t1_all, prefix='t1_')
        # plot_xzy(x_all, z_all, t2_all, prefix='t2_')
        # breakpoint()
        
    t1_all, t2_all, y_all, x_all, z_all, s_all, gt_s_all = get_pred(two_stage_model, test_loader,cost)
    print('average defferal to f2:', torch.mean(s_all))
    print('average ground truth defferal to f2:', torch.mean(gt_s_all))
    plot_xzy(x_all, z_all,s_all, prefix=str(cost)+'_s_')
    plot_xzy(x_all, z_all,gt_s_all, prefix=str(cost)+'_gt_s_')
    plot_xzy(x_all, z_all, y_all[:,1], prefix=str(cost)+'_gt_')
    plot_xzy(x_all, z_all, t1_all, prefix=str(cost)+'_t1_')
    plot_xzy(x_all, z_all, t2_all, prefix=str(cost)+'_t2_')

    
    training_log_dict['param_cs'] = cs
    training_log_dict['param_ds'] = ds
    training_log_dict['track_t1_acc'] = track_t1_acc
    training_log_dict['track_t2_acc'] = track_t2_acc
    training_log_dict['ls'] = ls
    training_log_dict['f1ls'] = f1ls
    training_log_dict['f2ls'] = f2ls
    training_log_dict['l01cs'] = l01cs


    return two_stage_model, training_log_dict, 


def sep_hinge_experiment(train_loader, test_loader, cost, two_stage_model, training_configs):
    
    epoch = training_configs['epoch']
    batch_size = training_configs['batch_size']
    lr = training_configs['lr']
    
    optimizer = torch.optim.Adam(two_stage_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    track_batch_loss = []
    track_epoch_loss = []
    track_t1_acc = []
    track_t2_acc = []
    training_log_dict = {}
    cs = []
    ds = []
    f1ls = []
    f2ls = []
    ls = []
    l01cs = []
    last_batch=None
    for i in range(epoch):
        running_loss = 0
        debug=False
        

        
        for i, (x_batch, z_batch, y_batch) in enumerate(tqdm(train_loader)):
            
            
            if  i%32 == 0:
                test_acc_t1, test_acc_t2 = compute_accuracies(two_stage_model, test_loader)
                track_t1_acc.append(test_acc_t1)
                track_t2_acc.append(test_acc_t2)
            
           
            optimizer.zero_grad()
            t1, t2, s, param_dict= two_stage_model(x_batch, z_batch, debug=debug)
            # s = 1-t1.argmax(0)
            s = 1- torch.abs(t1)
            # breakpoint()
            if 'c' in param_dict:

                cs.append(param_dict['c'].detach().numpy().item())
                ds.append(param_dict['d'].detach().numpy().item())
            debug=False
            #s = 1
            loss_f1, loss_f2 = sep_hinge(x_batch, z_batch, y_batch, cost, t1, t2, s)
            f1ls.append(loss_f1.detach().numpy().item()/x_batch.shape[0])
            f2ls.append(loss_f2.detach().numpy().item()/x_batch.shape[0])
            loss = l01c(t1, t2, y_batch, s, cost)['l01c loss']
            ls.append(loss.detach().numpy().item()/x_batch.shape[0])
            l01cs.append(loss.detach().numpy().item()/x_batch.shape[0])
            loss_f1.backward()
            loss_f2.backward()
            
            optimizer.step()

            running_loss += loss.item()
            track_batch_loss.append(loss.item())
            

        last_batch=x_batch.clone()
        avg_loss = running_loss/ len(train_loader.dataset)
        track_epoch_loss.append(avg_loss)
        print(f"Epoch {i+1}/{epoch}, Loss: {avg_loss}")
        scheduler.step()
        # t1_all, t2_all, y_all, x_all, z_all, s_all, gt_s_all = get_pred(two_stage_model, x_test ,z_test,y_test,E_max_py_xz, max_y_x, py_xz,cost, batch_size, test_n)
        
        # plot_xzy(x_all, z_all,s_all, prefix='s_')
        # plot_xzy(x_all, z_all, y_all[:,1], prefix='gt_')
        # plot_xzy(x_all, z_all, t1_all, prefix='t1_')
        # plot_xzy(x_all, z_all, t2_all, prefix='t2_')
        # breakpoint()
        
    t1_all, t2_all, y_all, x_all, z_all, s_all, gt_s_all = get_pred(two_stage_model, test_loader,cost)
    print('average defferal to f2:', torch.mean(s_all))
    print('average ground truth defferal to f2:', torch.mean(gt_s_all))
    plot_xzy(x_all, z_all,s_all, prefix=str(cost)+'_s_')
    plot_xzy(x_all, z_all,gt_s_all, prefix=str(cost)+'_gt_s_')
    plot_xzy(x_all, z_all, y_all[:,1], prefix=str(cost)+'_gt_')
    plot_xzy(x_all, z_all, t1_all, prefix=str(cost)+'_t1_')
    plot_xzy(x_all, z_all, t2_all, prefix=str(cost)+'_t2_')

    
    training_log_dict['param_cs'] = cs
    training_log_dict['param_ds'] = ds
    training_log_dict['track_t1_acc'] = track_t1_acc
    training_log_dict['track_t2_acc'] = track_t2_acc
    training_log_dict['ls'] = ls
    training_log_dict['f1ls'] = f1ls
    training_log_dict['f2ls'] = f2ls
    training_log_dict['l01cs'] = l01cs

    return two_stage_model, training_log_dict, 