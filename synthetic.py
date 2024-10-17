import torch
import numpy as np
import random

seed = 42  # or any number you choose

# Python random seed
random.seed(seed)

# NumPy random seed
np.random.seed(seed)

# PyTorch random seed
torch.manual_seed(seed)

# If you are using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

# Ensure deterministic behavior in PyTorch (optional)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from matplotlib import pyplot as plt
import torch
import sklearn
from sklearn.neural_network import MLPClassifier as MLP
import numpy as np
from tqdm import tqdm
import numpy as np
from scipy.special import expit 
import matplotlib.pyplot as plt

from generate_data import create_data, expected_max_p_y_given_xz, posterior_p_y_given_x, posterior_p_y_given_x_z

# y1 = MLP()
# y2 = MLP()
# s = MLP()





class simple(torch.nn.Module):
    def __init__(self, ):
        super(simple, self).__init__()
        self.params = []
        for i in range(5):
            self.params.append(torch.zeros((1,1), dtype=float, requires_grad=True))

        self.params.append(torch.zeros((1), dtype=float, requires_grad=True))
        self.params.append(torch.zeros((1), dtype=float, requires_grad=True))

        self.sigmoid = torch.nn.Sigmoid()
        # self.softmax = torch.nn.Softmax()
    def parameters(self):
        for param in self.params:
            yield param
        # return self.params
    def forward(self, x,z):
        y1 = (self.sigmoid(x*self.params[0] + self.params[1]))
        y2 = (self.sigmoid(x*self.params[2] + z*self.params[3] + self.params[4]))
        s = self.sigmoid(x-self.params[5])*self.sigmoid(self.params[6]-x)
        y1 = torch.cat((1-y1, y1), dim=1)
        y2 = torch.cat((1-y2, y2), dim=1)
        s = torch.cat((1-s, s), dim=1)
        # print(s)
        # print(y1.shape, y2.shape, s.shape)
        return y1, y2, s


class linear(torch.nn.Module):
    def __init__(self, feat_dim=1, out_dim=2, hidden_dim=100):
        super(linear, self).__init__()
        self.hidden_dim=hidden_dim
        self.y1 = torch.nn.Linear(feat_dim, out_dim)

        self.y2 = torch.nn.Linear(feat_dim, out_dim)

        self.s = torch.nn.Linear(feat_dim, out_dim)

        # self.y1_in = torch.nn.Linear(feat_dim, hidden_dim)
        # self.y1_out = torch.nn.Linear(hidden_dim, out_dim+1)
        # self.y2_in = torch.nn.Linear(feat_dim, hidden_dim)
        # self.y2_out = torch.nn.Linear(hidden_dim, out_dim)
        # self.relu = torch.nn.ReLU()

        self.y1_in = torch.nn.Linear(feat_dim, hidden_dim)
        self.y1_out = torch.nn.Linear(hidden_dim, out_dim)
        self.y2_in = torch.nn.Linear(feat_dim, hidden_dim)
        self.y2_out = torch.nn.Linear(hidden_dim, out_dim)
        self.relu = torch.nn.ReLU()

        self.s_in = torch.nn.Linear(feat_dim, hidden_dim)
        self.s_out = torch.nn.Linear(hidden_dim, out_dim)

        self.parameters()

        self.Softmax = torch.nn.Softmax()
    def forward(self, x,z):
        

        y1 = self.Softmax(self.y1_out(self.relu(self.y1_in(x))))
        y2 = self.Softmax(self.y2_out(self.relu(self.y2_in(x+z))))
        # y2 = self.Softmax(self.y2_out(self.relu(self.y2_in(torch.cat((x,z), dim=1)))))
        s = self.Softmax(self.s_out(self.relu(self.s_in(x))))

        # y1 = self.Softmax(self.y1(x))
        # y2 = self.Softmax(self.y2(x+z))
        # s = self.Softmax(self.s(x))

        # s = y1[:,-1].reshape((len(y1), 1))
        # y1 = y1[:, :-1]
        
        # print(y1.shape)
        # return y1, y2, s
        return torch.zeros((32,2)), y2, torch.zeros((32,2))

class model(torch.nn.Module):

    def __init__(self, feat_dim=1, out_dim=2, hidden_dim=1000):
        super(model, self).__init__()
        self.hidden_dim=hidden_dim
        self.y1_in = torch.nn.Linear(feat_dim, self.hidden_dim)
        self.y1_out = torch.nn.Linear(self.hidden_dim, out_dim)
        self.act = torch.nn.ReLU()
        self.y2_in = torch.nn.Linear(feat_dim*2, 2*self.hidden_dim)
        self.y2_out = torch.nn.Linear(2*self.hidden_dim, out_dim)
        self.s_in = torch.nn.Linear(feat_dim*2, 2*self.hidden_dim)
        self.s_in = torch.nn.Linear(feat_dim, self.hidden_dim)
        self.s_out = torch.nn.Linear(self.hidden_dim, 2)
        self.Softmax = torch.nn.Softmax()

    def forward(self, x,z):
        y1_in = self.act(self.y1_in(x))
        y1_out = self.Softmax((self.y1_out(y1_in)))
        y2_in = self.act(self.y2_in(torch.concatenate((x,z), dim=1)))
        y2_out = self.Softmax((self.y2_out(y2_in)))
        s_in = self.act(self.s_in(x))
        s_out = self.Softmax((self.s_out(s_in)))

        # if s.argmax() == 1:
        #     return y2_out
        return y1_out, y2_out, s_out
        
class mlp(torch.nn.Module):
    def __init__(self, feat_dim=1, out_dim=2, hidden_dim=100):
        super(mlp, self).__init__()
        self.hidden_dim=hidden_dim
        self.feat_dim = feat_dim
        self.y1_in = torch.nn.Linear(feat_dim, self.hidden_dim)
        self.y1_out = torch.nn.Linear(self.hidden_dim, out_dim)
        self.act = torch.nn.ReLU()
        self.Softmax = torch.nn.Softmax()
    def forward(self, x):
    
        y1_in = self.act(self.y1_in(x))
        y1_out = self.Softmax(self.act(self.y1_out(y1_in)))
        return y1_out

def risk(y1, y2, s,y, cost):
    # out = torch.ones(len(y1))*3
    risk = 0
    for i in range(len(y1)):
        if s[i]> 0.5:
            if int((y2[i].argmax() != y[i].argmax())):
                risk += 1
        else:
            if int(y1[i].argmax() != y[i].argmax()):
                risk += 1 - cost
    return risk/len(y)

def loss_fn(x,z,y,cost,y1,y2, K=2, joint=True, CE=False):
   
    # breakpoint()
    # r = s(torch.cat((p1,x), dim=1))[:,1]
    # breakpoint()
    # term1 = torch.log(torch.pow(1-r, cost*2+1)*r)

    ## terms are grouped:
    ##  termy: items which depend on the value of y
    ##  termr: items which depend on p_r
    ##  termsum: the sum item
    # print(y)
    if CE:
        p1, p2, s = y1(x,z)
        # print('returning CE')
        return torch.nn.CrossEntropyLoss(reduction='sum')(p2,y)
    if joint:
        p1, p2, s = y1(x,z)
        # termy = torch.zeros((len(p1)))
        # # breakpoint()
        # for k in range(len(p1)):
        #     # termy[k] = -torch.log(p1[k,int(y[k])]) -torch.log(p2[k, int(y[k])]) 
        #     termy[k] = -torch.log(p1[k,np.argmax(y[k])]) -torch.log(p2[k, np.argmax(y[k])]) 

        # termy = torch.nn.CrossEntropyLoss()(p2,y) + torch.nn.CrossEntropyLoss()(p1[:, :-1], y)
        # termy = torch.nn.CrossEntropyLoss(reduction='sum')(p2,y) + torch.nn.CrossEntropyLoss(reduction='sum')(p1, y)
        termy = 0


        for i in range(len(y)):
            # a = ys.argmax()
            # print(ys)
            termy += -torch.log(p1[i,int(y[i,1])]) - torch.log(p2[i,int(y[i,1])])
            # print(p1[int(ys[1])], p2[int(ys[1])])


        if s.shape[1] != 0:
            termr = -torch.log(s[:,-1]) - (1+cost*(len(p1[0])))*torch.log(1-s[:,-1])
        else:
            termr = -torch.log(s) - (1+cost*(len(p1[0])))*torch.log(1-s)

        termr = torch.sum(termr)
        termsum = torch.zeros(len(p1))

        for k in range(len(p1[0])):
            termsum += torch.log(p1[:, k])

        termsum *= -cost
        termsum = torch.sum(termsum)
        # print(torch.sum(termr), torch.sum(termsum), torch.sum(termy))
        # return torch.sum(termr + termsum + termy)

        # print(termy, termr, termsum)
        return termy + termr + termsum

        # return torch.sum(termr+termy)
        # return torch.sum(termy)
        # breakpoint()
        # return torch.sum(p1.argmax(dim=1)-y)

        # return torch.sum(torch.nn.CrossEntropyLoss()(p1[:, :-1],y[:, -1]))
        # return  torch.nn.CrossEntropyLoss()(p2,y) + torch.nn.CrossEntropyLoss()(p1[:, :-1], y)
        # return 
    print('uh oh')
    p1 = y1(x)
    p2 = y2(torch.cat((x,z), dim=1))
    
    termy = torch.zeros((len(p1)))
    for k in range(len(p1)):
        termy[k] = -torch.log(p1[:,int(y[k])]) -torch.log(p2[:, int(y[k])]) 

    termr = -torch.log(p1[:,-1]) - (1+cost*(len(p1[0])-1))*torch.log(1-p1[:,-1])

    termsum = torch.zeros(len(p1[0])-1)

    for k in range(len(p1[0])-1):
        termsum += torch.log(p1[:, k])
    termsum *= -cost
    # term3 = torch.zeros((len(p1)))
    # for k in range(len(p1)):
    #     term3[k] = -torch.log(p1[:,int(y[k])]) -torch.log(p2[:, int(y[k])])
    return torch.sum(torch.stack([termr, termsum, termy]))

def eval(x,z,y,cost,y1,y2, joint=True, debug=False):
    
    if joint:
        p1, p2, s = y1(x,z)
        # print(p1, p2, s)
        
    else:
        p1 = y1(x)
        p2 = y2(torch.cat((x,z), dim=1))
        

    # breakpoint()
    # r = s(torch.cat((p1,x), dim=1))
    # breakpoint()
    # r = p1[:,-1]
    # print(r)
    # breakpoint()
    # p1 = p1[:,:-1]
    if s.shape[1] != 0:
        r = s[:, -1]
    else:
        r = s

    # r = p1[:,-1]
    # p1 = p1[:, :-1]

    # num_y1 = torch.sum(torch.where(torch.argmax(p1, dim=1) == y.reshape(1,len(y)),1, 0))
    num_y1 = torch.sum(torch.where(torch.argmax(p1, dim=1) == y.argmax(dim=1),1, 0))
    num_y2 = torch.sum(torch.where(torch.argmax(p2, dim=1) == y.argmax(dim=1),1, 0))

    # num_y2 = torch.sum(torch.where(torch.argmax(p2, dim=1) == y.reshape(1,len(y)), 1, 0))
    # breakpoint()
    num_bad_rs = 0
    num_good_nors = 0
    num_bad_nors = 0
    num_good_rs = 0
    model_out = []
    if joint: 
        for i in range(len(p1)):
            # if torch.argmax(p1[i]) == int(y[i]):
            if torch.argmax(p1[i]) == np.argmax(y[i]):

                # if torch.max(p1[i]) < r[i]:
                if r[i] > 0.5:
                    num_bad_rs += 1
                else:
                    num_good_nors += 1
            else:
                if torch.max(p1[i]) < r[i]:
                    num_good_rs += 1
                else:
                    num_bad_nors += 1
    else:
        for i in range(len(p1)):
            if torch.argmax(p1[i]) == int(y[i]):
                # if torch.argmax(r[i]) == 1:
                if r[i] > 0.5:
                    num_bad_rs += 1
                else:
                    num_good_nors += 1
            else:
                if torch.argmax(r[i])== 1:
                    num_good_rs += 1
                else:
                    num_bad_nors += 1
        assert len(p1) == len(p2)
    # if debug:
    #     breakpoint()
    if s.shape[1] != 0:
        return {'acc y1': num_y1/len(p1), 'acc y2': num_y2/len(p2), 'good reject': num_good_rs/len(p1), 'bad reject': num_bad_rs/len(p1), 'good non-reject': num_good_nors/len(p1), 'bad non-reject': num_bad_nors/len(p1)}, {'p1': p1, 'p2': p2, 's': s[:, -1], 'x': x, 'y': y, 'z':z}
    else:
        return {'acc y1': num_y1/len(p1), 'acc y2': num_y2/len(p2), 'good reject': num_good_rs/len(p1), 'bad reject': num_bad_rs/len(p1), 'good non-reject': num_good_nors/len(p1), 'bad non-reject': num_bad_nors/len(p1)}, {'p1': p1, 'p2': p2, 's': s, 'x': x, 'y': y, 'z':z}

# model = model()

def exp(data_dict, cost=0.01):
    train_d = data_dict['train_d']
    test_d = data_dict['test_d']

    x_train = data_dict['x']
    y_train = data_dict['y']
    z_train = data_dict['z']
    
    x_val = data_dict['x_val']
    y_val = data_dict['y_val']
    z_val = data_dict['z_val']
    
    x_test = data_dict['x_test']
    y_test = data_dict['y_test']
    z_test = data_dict['z_test']
    

    jm = simple()
  

    epoch=1
    batch_size=32
    optimizer = torch.optim.Adam(jm.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

   
    running_loss = 0
    from tqdm import tqdm
    last_3_valid = torch.zeros(3)
    early_stopper = EarlyStopper(patience=15, min_delta=5)

    
    for i in range(epoch):
        for batch in tqdm(range(batch_size, train_d+batch_size, batch_size)):

            x_batch = x_train[batch-batch_size:batch]
            z_batch = z_train[batch-batch_size:batch]
            y_batch = y_train[batch-batch_size:batch]

            optimizer.zero_grad()


            # loss = loss_fn(x_batch, z_batch, y_batch, cost, y1, y2)
            # loss = loss_fn(x_batch, z_batch, y_batch, cost, jm, None)
            loss = loss_fn(x_batch, z_batch, y_batch, cost, jm, None, CE=False)

            loss.backward()
            # for i,p in enumerate(jm.parameters()):
            #     print(i, p.grad.norm())
            optimizer.step()

            running_loss += loss.item()
            if batch % (batch_size*10) == 0:
                # last_loss = running_loss / 320 # loss per batch
                # print('  batch {} loss: {}'.format(batch/batch_size, loss))
              
                valid, ___ = eval(x_val[batch-batch_size:batch], z_val[batch-batch_size:batch], y_val[batch-batch_size:batch], cost, jm,None)
                # print(valid)
                x_batch = x_val[batch-batch_size:batch]
                z_batch = z_val[batch-batch_size:batch]
                y_batch = y_val[batch-batch_size:batch]
                valid_loss = loss = loss_fn(x_batch, z_batch, y_batch, cost, jm, None, CE=False)
                if early_stopper.early_stop(valid_loss):             
                    print('early stopping')
                    break
            if batch % (batch_size*100) == 0:

                scheduler.step()
    valid, ___ = eval(x_val[batch-batch_size:batch], z_val[batch-batch_size:batch], y_val[batch-batch_size:batch], cost, jm,None)

    result = {}
    # preds = torch.zeros((0, 6))
    p1s = []
    p2s = []
    ss = []
    xs = []
    ys = []
    zs = []
    for key in valid.keys():
        result[key] = []
    for batch in range(batch_size, test_d+batch_size, batch_size):

        x_batch = x_test[batch-batch_size:batch]
        z_batch = z_test[batch-batch_size:batch]
        y_batch = y_test[batch-batch_size:batch]


        out, pred = eval(x_batch, z_batch, y_batch, cost, jm, None, debug=True)
        # preds = torch.concatenate((preds, pred))
        p1s += list(pred['p1'])
        p2s += list(pred['p2'])
        ss += list(pred['s'])
        xs += list(pred['x'])
        ys += list(pred['y'])
        zs += list(pred['z'])
        for key, value in out.items():
            result[key].append(value)

    mean_result = {}
    sdd_result = {}
    for key, value in result.items():
        mean_result[key] = np.mean(value)
        sdd_result[key] = np.std(value)
    # print(start_params==list(y1.parameters()) + list(y2.parameters()))
    # breakpoint()
    # for param in jm.parameters():
    #     print(param)
    return mean_result, sdd_result, {'p1': p1s, 'p2': p2s, 's': ss, 'x': xs, 'y': ys, 'z': zs}

if __name__ == '__main__':
    costs = [0, 0.002, 0.004, 0.006, 0.008, 0.01,0.05, 0.1]


    num_accept = []
    num_reject = []
    accept_acc = []
    reject_acc = []
    y1_acc = []
    y2_acc = []
    l_sdd = []
    risks = []
    pick_1 = []
    pick_2 = []
    t_risks = []
    t_x_acc = []
    t_xz_acc = []


    data_dict = create_data(trial = 2,train_d = 32*10000, test_d = 32*1000)
    for cost in tqdm(costs):
        result, sdd, preds = exp(data_dict, cost=cost)
        l_sdd.append(sdd)
        for key, value in result.items():
            try:
                result[key] = torch.round(value.items(), decimals=2)
            except:
                result[key] = np.round(value, decimals=2)
        num_accept.append(result['good non-reject'] + result['bad non-reject'])
        accept_acc.append(str(result['good non-reject']) + '/' + str(result['bad non-reject']))
        num_reject.append(result['good reject'] + result['bad reject'])
        reject_acc.append(str(result['good reject']) + '/' + str(result['bad reject']))
        y1_acc.append(result['acc y1'])
        y2_acc.append(result['acc y2'])
        """
        colors = []
        for p in preds['s']:
            if p <= 0.5:
                colors.append('r')
            else:
                colors.append('g')
        fig1, ax1 = plt.subplots()
        # print(len(preds['x']))
        ax1.scatter(torch.stack(preds['x']), torch.stack(preds['z']), color = colors, s=3)
        ax1.set_title('Cost = ' + str(cost))
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='r', label='s < 0.5')
        plt.legend(handles=[red_patch])
        green_patch = mpatches.Patch(color='g', label='s > 0.5')
        ax1.legend(handles=[red_patch, green_patch])
        fig1.savefig('./' + str(cost)+'.pdf', format='pdf')
        """
        
        risks.append(risk(preds['p1'], preds['p2'], preds['s'], preds['y'], cost))
        pick_1.append(torch.mean(torch.where(torch.stack(preds['p1']).argmax(dim=1) == torch.stack(preds['y']).argmax(dim=1), 1-cost, 0), dtype=float))
        pick_2.append(torch.mean(torch.where(torch.stack(preds['p2']).argmax(dim=1) == torch.stack(preds['y']).argmax(dim=1), 1, 0), dtype=float))
        # ax1.legend()
        binary_map = np.zeros_like(torch.stack(preds['x']))
        x_values = torch.stack(preds['x'])
        z_values = torch.stack(preds['z'])
        y_values = torch.stack(preds['y'])
        t_risk = 0
        t_x = 0
        t_xz = 0
        for i in range(len(x_values)):
            x = x_values[i]
        
            # Compute the expected max p(y | x, z) over z
            expected_max_prob = data_dict['test_E'][i]
            xz_post = data_dict['test_py_xz'][i] 
            #check = posterior_p_y_given_x_z(x, z_values[i])
            # Compute the marginalized max_y p(y | x)
            max_prob = data_dict['test_max'][i] #posterior_p_y_given_x(x)
            
            # Compute the difference
            diff = expected_max_prob - max_prob
            
            binary_map[i] = torch.where(diff > cost, 1, 0)
            
            if np.where(xz_post > 0.5, 1, 0) == y_values[i].argmax():
                t_xz += 1
            if np.where(max_prob > 0.5, 1, 0) == y_values[i].argmax():
                t_x += 1
        

        # t_risk = 0
        # for i in range(len(binary_map)):
            if binary_map[i] == 1:
                if np.where(xz_post > 0.5, 1, 0).item() != y_values[i].argmax().item():
            #     if torch.bernoulli(sigmoid(x_values[i] + z_values[i])) != y_values[i].argmax():
                    t_risk += 1
            elif binary_map[i] == 0:
                if np.where(max_prob > 0.5, 1, 0).item() != y_values[i].argmax().item():
            #     if torch.bernoulli(sigmoid(x_values[i])) != y_values[i].argmax():
                    t_risk += 1-cost
                    
                    
        t_risk /= len(binary_map)
        t_risks.append(t_risk)
        t_xz_acc.append(t_xz/len(binary_map))
        t_x_acc.append(t_x/len(binary_map))
        # breakpoint()
    # colors = []
    # for p in preds['y']:
    #         if p[1] == 1:
    #             colors.append('r')
    #         else:
    #             colors.append('g')
    # ax1.scatter(torch.stack(preds['x']), torch.stack(preds['z']), color = colors, s=3)
    # ax1.set_title('cringe')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Z')

    fig1, ax1 = plt.subplots()

    ax1.plot(costs, y2_acc)
    ax1.set_ylim(0, 1)
    ax1.set_title("Y2 Accuracy (% of all predictions) vs. Cost")

    fig1, ax1 = plt.subplots()

    ax1.plot(costs, y1_acc)
    ax1.set_ylim(0, 1)
    ax1.set_title("Y1 Accuracy (% of all predictions) vs. Cost")

    fig1, ax1 = plt.subplots()

    ax1.plot(costs, num_accept)
    ax1.set_title("Number of Y1-Accepts (% of all predictions) vs Cost")
    ax1.set_ylim(0, 1)
    # for i in range(len(accept_acc)):
    #     # if i != 0 and np.abs(num_accept[i-1]-num_accept[i]) < 0.1:
    #     #     plt.text(costs[i]+0.05, num_accept[i]+0.05, accept_acc[i])
    #     # else:
    #     #     plt.text(costs[i], num_accept[i], accept_acc[i])
    #     plt.text(costs[i], num_accept[i], accept_acc[i])
    # plt.text(6, 0.7, "Note: we show at each point \n [Correct Accepts]/[Incorrect Accepts] \n based on whether Y1 was correct")

    fig1, ax1 = plt.subplots()
    ax1.plot(costs, num_reject)
    ax1.set_title("Number of Y1-Rejects (% of all predictions) vs Cost ")
    ax1.set_ylim(0, 1)

    fig1, ax1 = plt.subplots()
    ax1.plot(costs, risks, label='Ours')
    # ax1.plot(costs, pick_1, label='Aways Pick f1')
    # ax1.plot(costs, pick_2, label='Alwyays Pick f2')
    ax1.plot(costs, t_risks, label=r'$\tilde{f}^*$')
    ax1.legend()
    ax1.set_title("Risk Loss vs Cost")
    ax1.set_ylabel("Risk")
    ax1.set_xlabel("Cost")

    fig1.savefig('riscfig.pdf', format='pdf')

    fig1, ax1 = plt.subplots()
    ax1.plot(costs, t_x_acc, label='x acc')
    ax1.plot(costs, t_xz_acc, label='xz acc')
    ax1.legend()

    fig1.savefig('post_acc.pdf', format='pdf')

    # ax1.set_ylim(0,1)
    # for i in range(len(reject_acc)):
    #     # if i != 0 and np.abs(num_reject[i-1]-num_reject[i]) < 0.1:
    #     #     plt.text(costs[i]+0.05, num_reject[i]+0.05, reject_acc[i])
    #     # else:
    #     #     plt.text(costs[i], num_reject[i], reject_acc[i])
    #     plt.text(costs[i], num_reject[i], reject_acc[i])
    # plt.text(6, 0.7, "Note: we show at each point \n [Correct Rejects]/[Incorrect rejects] \n based on whether Y1 was correct")
    # d = 1000
    # x = torch.normal(mean=torch.zeros((d,1)), std=torch.ones((d,1)))
    # z = torch.normal(mean=torch.zeros((d,1)), std=torch.ones((d,1)))

    # y = torch.ones((d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x*z))

    # sxz = torch.nn.Sigmoid()(x*z)

    # y, ___ = torch.sort(y)

    # sxz, ___ = torch.sort(sxz)


    # fig2, ax2 = plt.subplots()


    # ax2.hist(sxz[:-1], bins=5)
