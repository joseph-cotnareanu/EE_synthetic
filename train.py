import torch 
from tqdm import tqdm
from training.loss import loss_hinge_joint
from matplotlib import pyplot as plt

def train_two_stage_experiment(data_dict, cost, two_stage_model):
    
    epoch = 50
    batch_size = 32
    optimizer = torch.optim.Adam(two_stage_model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    train_n = data_dict['train_n']
    test_n = data_dict['test_n']
    train_n = 2045

    x_train = data_dict['x_train']
    y_train = data_dict['y_train']
    z_train = data_dict['z_train']
    
    x_val = data_dict['x_val']
    y_val = data_dict['y_val']
    z_val = data_dict['z_val']
    track_batch_loss = []
    track_epoch_loss = []
    track_val_loss = []
    training_log_dict = {}
    cs = []
    for i in range(epoch):
        running_loss = 0
        debug=False
        for batch in tqdm(range(batch_size, train_n+batch_size, batch_size)):
            x_batch = x_train[batch-batch_size:batch]
            z_batch = z_train[batch-batch_size:batch]
            y_batch = y_train[batch-batch_size:batch]

            optimizer.zero_grad()
            t1, t2, s, c = two_stage_model(x_batch, z_batch, debug=debug)
            cs.append(c.detach().numpy().item())
            debug=False
            #s = 1
            loss = loss_hinge_joint(x_batch, z_batch, y_batch, cost, t1, t2, s)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            track_batch_loss.append(loss.item())
           

           

        avg_loss = running_loss / train_n
        track_epoch_loss.append(avg_loss)
        print(f"Epoch {i+1}/{epoch}, Loss: {avg_loss}")
        scheduler.step()
        
    fig1, ax1 = plt.subplots()
    ax1.plot(cs)
    fig1.savefig('/root/EE_synthetic/cplot.png')
    return two_stage_model, training_log_dict, 
