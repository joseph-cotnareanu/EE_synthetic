from matplotlib import pyplot as plt
import torch
import sklearn
from sklearn.neural_network import MLPClassifier as MLP
import numpy as np

# y1 = MLP()
# y2 = MLP()
# s = MLP()



class linear(torch.nn.Module):
    def __init__(self, feat_dim=1, out_dim=2, hidden_dim=100):
        super(linear, self).__init__()
        self.hidden_dim=hidden_dim
        self.y1 = torch.nn.Linear(feat_dim, out_dim + 1)

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
        self.s_out = torch.nn.Linear(hidden_dim, 1)

        self.y2 = torch.nn.Linear(feat_dim*2, out_dim)
        self.y2 = torch.nn.Linear(feat_dim, out_dim)

        self.Softmax = torch.nn.Softmax()
    def forward(self, x,z):
        # y1 = self.Softmax(self.y1(x))
        # # y2 = self.Softmax(self.y2(torch.concatenate((x,z), dim=1)))
        # y2 = self.Softmax(self.y2(x+z))

        y1 = self.Softmax(self.y1_out(self.relu(self.y1_in(x))))
        y2 = self.Softmax(self.y2_out(self.relu(self.y2_in(x+z))))

        # s = y1[:,-1].reshape((len(y1), 1))
        # y1 = y1[:, :-1]
        s = self.relu(self.s_out(self.relu(self.s_in(x))))
        # print(y1.shape)
        return y1, y2, s

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
        print('returning CE')
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



        termr = -torch.log(s[:,-1]) - (1+cost*(len(p1[0])))*torch.log(1-s[:,-1])
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

    r = s[:, -1]

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
    if debug:
        breakpoint()
    return {'acc y1': num_y1/len(p1), 'acc y2': num_y2/len(p2), 'good reject': num_good_rs/len(p1), 'bad reject': num_bad_rs/len(p1), 'good non-reject': num_good_nors/len(p1), 'bad non-reject': num_bad_nors/len(p1)}, {'p1': p1, 'p2': p2, 's': s[:, -1], 'x': x, 'y': y, 'z':z}
# model = model()

def exp(train_d=32*20, test_d=32*20, cost=0.01):


    # d = 32*20
    # cost=0.01
    # cost = cost

    c = torch.ones(train_d)*cost

    # x = torch.normal(mean=torch.zeros((1, d)), std=torch.ones((1, d)))
    # z = torch.normal(mean=torch.zeros((1, d)), std=torch.ones((1, d)))

    x = torch.normal(mean=torch.zeros((train_d,1)), std=torch.ones((train_d,1)))
    z = torch.normal(mean=torch.zeros((train_d,1)), std=torch.ones((train_d,1)))

    y = (torch.ones((train_d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x+z))).type(torch.LongTensor)
    new_y = torch.zeros((train_d, 2))
    for i in range(len(y)):
        if y[i] == 0:
            new_y[i] = torch.tensor([1,0])
        elif y[i] == 1:
            new_y[i] = torch.tensor([0,1])

    y = new_y

    # y = torch.ones((d, 2))
    # for i in range(len(x)):
    #     y[int(torch.bernoulli(torch.nn.Sigmoid()(x[i]+z[i])))] = 1
    #     y[int(1-torch.bernoulli(torch.nn.Sigmoid()(x[i]+z[i])))] = 0
    # y = torch.ones((d, 1))

    x_val = torch.normal(mean=torch.zeros((train_d,1)), std=torch.ones((train_d,1)))
    z_val = torch.normal(mean=torch.zeros((train_d,1)), std=torch.ones((train_d,1)))

    y_val = torch.ones((train_d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x_val+z_val))

    new_y = torch.zeros((train_d, 2))
    for i in range(len(y_val)):
        if y_val[i] == 0:
            new_y[i] = torch.tensor([1,0])
        elif y_val[i] == 1:
            new_y[i] = torch.tensor([0,1])

    y_val = new_y
    # y_val = torch.ones((d, 2))
    # for i in range(len(x)):
    #     y_val[int(torch.bernoulli(torch.nn.Sigmoid()(x_val[i]+z_val[i])))] = 1
    #     y_val[int(1-torch.bernoulli(torch.nn.Sigmoid()(x_val[i]+z_val[i])))] = 0

    x_test = torch.normal(mean=torch.zeros((test_d,1)), std=torch.ones((test_d,1)))
    z_test = torch.normal(mean=torch.zeros((test_d,1)), std=torch.ones((test_d,1)))
    # print('l. 280', len(x_test))
    y_test = torch.ones((test_d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x_test+z_test))
    # print('l. 282', len(y_test))
    new_y = torch.zeros((test_d, 2))
    for i in range(len(y_test)):
        if y_test[i] == 0:
            new_y[i] = torch.tensor([1,0])
        elif y_test[i] == 1:
            new_y[i] = torch.tensor([0,1])

    y_test = new_y
    # print('l. 291', len(y_test))



    # y_test = torch.ones((d, 1))
    # y_test = torch.ones((d, 2))
    # for i in range(len(x)):
    #     y_test[int(torch.bernoulli(torch.nn.Sigmoid()(x_test[i]+z_test[i])).item())] = 1
    #     y_test[int(1-torch.bernoulli(torch.nn.Sigmoid()(x_test[i]+z_test[i])).item())] = 0


    y1 = mlp(feat_dim=1, out_dim=3, hidden_dim=100)
    y2 = mlp(feat_dim=2, out_dim=2, hidden_dim=200)

    jm = linear(feat_dim=1, out_dim=2, hidden_dim=100)
    # s = mlp(feat_dim=3, hidden_dim=200)


    epoch=1
    batch_size=32
    optimizer = torch.optim.Adam(jm.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # breakpoint()
    # start_params = list(y1.parameters().clone()) + list(y2.parameters().clone())
    running_loss = 0
    from tqdm import tqdm
    for i in range(epoch):
        for batch in tqdm(range(batch_size, train_d+batch_size, batch_size)):

            x_batch = x[batch-batch_size:batch]
            z_batch = z[batch-batch_size:batch]
            y_batch = y[batch-batch_size:batch]

            optimizer.zero_grad()


            # loss = loss_fn(x_batch, z_batch, y_batch, cost, y1, y2)
            # loss = loss_fn(x_batch, z_batch, y_batch, cost, jm, None)
            loss = loss_fn(x_batch, z_batch, y_batch, cost, jm, None, CE=False)

            loss.backward()
            # for i,p in enumerate(jm.parameters()):
            #     print(i, p.grad.norm())
            optimizer.step()

            running_loss += loss.item()
            if batch % 50 == 0:
                # last_loss = running_loss / 320 # loss per batch
                # print('  batch {} loss: {}'.format(batch/batch_size, loss))
              
                valid, ___ = eval(x_val[batch-batch_size:batch], z_val[batch-batch_size:batch], y_val[batch-batch_size:batch], cost, jm,None)
                # print(valid)
            # scheduler.step()
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
    return mean_result, sdd_result, {'p1': p1s, 'p2': p2s, 's': ss, 'x': xs, 'y': ys, 'z': zs}

# costs = [0.001, 0.002, 0.005, 0.1, 0.2, 0.5, 1, 2, 5, 10]
# costs = [0.01, 0.1, 1, 10, 1000]
# costs = [0, 0.1, 1, 10]
# costs = [0, 0]
# costs = list(np.arange(0, 1, 0.2))
# costs = np.arange()
# costs = [10000000]
# costs = [0.0000001]
# costs = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
# costs = [0, 0, 0,0,0]
# costs = [10000, 10000, 10000,10000]
# costs = np.arange(0, 0.1, 0.01)
costs = [0, 0.05, 0.1, 1, 1000]
# costs = np.arange(0, 1, 0.2)
num_accept = []
num_reject = []
accept_acc = []
reject_acc = []
y1_acc = []
y2_acc = []
l_sdd = []

for cost in costs:
    result, sdd, preds = exp(train_d = 32*10000, test_d = 32*1000,cost=cost)
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
    fig1.savefig('./' + str(cost)+'.png')
    # ax1.legend()

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


for i in range(len(reject_acc)):
    # if i != 0 and np.abs(num_reject[i-1]-num_reject[i]) < 0.1:
    #     plt.text(costs[i]+0.05, num_reject[i]+0.05, reject_acc[i])
    # else:
    #     plt.text(costs[i], num_reject[i], reject_acc[i])
    plt.text(costs[i], num_reject[i], reject_acc[i])
plt.text(6, 0.7, "Note: we show at each point \n [Correct Rejects]/[Incorrect rejects] \n based on whether Y1 was correct")
# d = 1000
# x = torch.normal(mean=torch.zeros((d,1)), std=torch.ones((d,1)))
# z = torch.normal(mean=torch.zeros((d,1)), std=torch.ones((d,1)))

# y = torch.ones((d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x*z))

# sxz = torch.nn.Sigmoid()(x*z)

# y, ___ = torch.sort(y)

# sxz, ___ = torch.sort(sxz)


# fig2, ax2 = plt.subplots()


# ax2.hist(sxz[:-1], bins=5)
