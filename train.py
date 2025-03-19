import torch 
from tqdm import tqdm
from hinge_utils import compute_accuracies
from training.loss import loss_hinge_joint
from matplotlib import pyplot as plt



def train_two_stage_experiment(data_dict, cost, two_stage_model):
    
    epoch = 2
    batch_size = 4
    optimizer = torch.optim.Adam(two_stage_model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
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
    for i in range(epoch):
        running_loss = 0
        debug=False

        
        for batch in tqdm(range(batch_size, train_n+batch_size, batch_size)):
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
    ax1.set_title('cplot')
    fig1.savefig('cplot.pdf')
    plt.close()
    fig1, ax1 = plt.subplots()
    ax1.plot(ds)
    ax1.set_title('dplot')
    fig1.savefig('dplot.pdf')
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].plot(track_t1_acc, label='Accuracy 1', marker='o')
    ax[0].set_title('f1')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    ax[1].plot(track_t2_acc, label='Accuracy 2', marker='o', color='r')
    ax[1].set_title('f2')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('acc.pdf')

    return two_stage_model, training_log_dict, 
