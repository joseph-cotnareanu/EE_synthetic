
import torch 
from tqdm import tqdm
from training.loss import loss_hinge_joint

def train_two_stage_experiment(data_dict, cost, two_stage_model):
    epoch=6
    batch_size=32
    optimizer = torch.optim.Adam(two_stage_model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

   
    train_n = data_dict['train_n']
    test_n = data_dict['test_n']

    x_train = data_dict['x_train']
    y_train = data_dict['y_train']
    z_train = data_dict['z_train']
    
    x_val = data_dict['x_val']
    y_val = data_dict['y_val']
    z_val = data_dict['z_val']
    
    training_log_dict = {}
    
    for i in range(epoch):
        running_loss = 0
        for batch in tqdm(range(batch_size, train_n+batch_size, batch_size)):

            x_batch = x_train[batch-batch_size:batch]
            z_batch = z_train[batch-batch_size:batch]
            y_batch = y_train[batch-batch_size:batch]

            optimizer.zero_grad()
            t1, t2, s = two_stage_model(x_batch, z_batch)
            loss = loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s)

            loss.backward()
            # for i,p in enumerate(jm.parameters()):
            #     print(i, p.grad.norm())
            optimizer.step()

            running_loss += loss.item()
            if batch % (batch_size*10) == 0:
                # last_loss = running_loss / 320 # loss per batch
                # print('  batch {} loss: {}'.format(batch/batch_size, loss))
              
                # valid, ___ = eval(x_val[batch-batch_size:batch], z_val[batch-batch_size:batch], y_val[batch-batch_size:batch], cost, jm,None)
                # print(valid)
                x_batch = x_val[batch-batch_size:batch]
                z_batch = z_val[batch-batch_size:batch]
                y_batch = y_val[batch-batch_size:batch]
                p1, p2, s = two_stage_model(x_batch, z_batch)
                valid_loss = loss = loss_hinge_joint(x_batch, z_batch, y_batch, cost, p1, p2, s)
                
                # if early_stopper.early_stop(valid_loss):             
                #     print('early stopping')
                #     break
            if batch % (batch_size*100) == 0:

                scheduler.step()
        print(running_loss/train_n)
    return two_stage_model, training_log_dict