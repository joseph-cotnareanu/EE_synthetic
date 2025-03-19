import torch 
from tqdm import tqdm
from hinge_utils import compute_accuracies, get_pred
from storing_plotting import plot_xzy
from training.loss import loss_hinge_joint




def train_two_stage_experiment(data_dict, cost, two_stage_model, training_configs):
    
    epoch = training_configs['epoch']
    batch_size = training_configs['batch_size']
    lr = training_configs['lr']
    
    optimizer = torch.optim.Adam(two_stage_model.parameters(), lr=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_n = data_dict['train_n']
    test_n = data_dict['test_n']
   

    x_train = data_dict['x_train']
    y_train = data_dict['y_train']
    z_train = data_dict['z_train']
    
    x_test = data_dict['x_test']
    y_test = data_dict['y_test']
    z_test = data_dict['z_test']
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
    for i in range(epoch):
        running_loss = 0
        debug=False

        
        for batch in tqdm(range(batch_size, train_n+batch_size, batch_size)):
            # test_acc_t1, test_q
            # test_acc_t1, test_acc_t2 = compute_accuracies(two_stage_model, x_test ,z_test,y_test,batch_size, test_n )
            # track_t1_acc.append(test_acc_t1)
            # track_t2_acc.append(test_acc_t2)
            # breakpoint()
            if (batch-batch_size) % 32 == 0:
                test_acc_t1, test_acc_t2 = compute_accuracies(two_stage_model, x_test ,z_test,y_test,batch_size, test_n )
                track_t1_acc.append(test_acc_t1)
                track_t2_acc.append(test_acc_t2)
                
            x_batch = x_train[batch-batch_size:batch]
            z_batch = z_train[batch-batch_size:batch]
            y_batch = y_train[batch-batch_size:batch]

            optimizer.zero_grad()
            t1, t2, s, c, d= two_stage_model(x_batch, z_batch, debug=debug)
            cs.append(c.detach().numpy().item())
            ds.append(d.detach().numpy().item())
            debug=False
            #s = 1
            loss, loss_f1, loss_f2 = loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s)
            f1ls.append(loss_f1.detach().numpy().item())
            f2ls.append(loss_f2.detach().numpy().item())
            ls.append(loss.detach().numpy().item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            track_batch_loss.append(loss.item())
            
        

        avg_loss = running_loss / train_n
        track_epoch_loss.append(avg_loss)
        print(f"Epoch {i+1}/{epoch}, Loss: {avg_loss}")
        scheduler.step()
        t1_all, t2_all, y_all, x_all, z_all = get_pred(two_stage_model, x_test ,z_test,y_test,batch_size, test_n)

        plot_xzy(x_all, z_all, y_all[:,1], prefix='gt')
        plot_xzy(x_all, z_all, t1_all, prefix='t1')
        plot_xzy(x_all, z_all, t2_all, prefix='t2')
        # breakpoint()
        
    t1_all, t2_all, y_all, x_all, z_all = get_pred(two_stage_model, x_test ,z_test,y_test,batch_size, test_n)
    plot_xzy(x_all, z_all, y_all[:,1], prefix='gt')
    plot_xzy(x_all, z_all, t1_all, prefix='t1')
    plot_xzy(x_all, z_all, t2_all, prefix='t2')

    
    training_log_dict['param_cs'] = cs
    training_log_dict['param_ds'] = ds
    training_log_dict['track_t1_acc'] = track_t1_acc
    training_log_dict['track_t2_acc'] = track_t2_acc
    training_log_dict['ls'] = ls
    training_log_dict['f1ls'] = f1ls
    training_log_dict['f2ls'] = f2ls

    return two_stage_model, training_log_dict, 
