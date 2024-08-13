from matplotlib import pyplot as plt
import torch
import sklearn
from sklearn.neural_network import MLPClassifier as MLP
import numpy as np

# y1 = MLP()
# y2 = MLP()
# s = MLP()





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

def loss_fn(x,z,y,cost,y1,y2, K=2, joint=True):
   
    # breakpoint()
    # r = s(torch.cat((p1,x), dim=1))[:,1]
    # breakpoint()
    # term1 = torch.log(torch.pow(1-r, cost*2+1)*r)

    ## terms are grouped:
    ##  termy: items which depend on the value of y
    ##  termr: items which depend on p_r
    ##  termsum: the sum item
    if joint:
        p1, p2, s = y1(x,z)
        # termy = torch.zeros((len(p1)))
        # # breakpoint()
        # for k in range(len(p1)):
        #     # termy[k] = -torch.log(p1[k,int(y[k])]) -torch.log(p2[k, int(y[k])]) 
        #     termy[k] = -torch.log(p1[k,np.argmax(y[k])]) -torch.log(p2[k, np.argmax(y[k])]) 

        # termy = torch.nn.CrossEntropyLoss()(p2,y) + torch.nn.CrossEntropyLoss()(p1[:, :-1], y)
        termy = torch.nn.CrossEntropyLoss()(p2,y) + torch.nn.CrossEntropyLoss()(p1, y)


        termr = -torch.log(s[:,-1]) - (1+cost*(len(p1[0])-1))*torch.log(1-s[:,-1])

        termsum = torch.zeros(len(p1))

        for k in range(len(p1[0])):
            termsum += torch.log(p1[:, k])

        termsum *= -cost
        print(torch.sum(termr), torch.sum(termsum), torch.sum(termy))
        return torch.sum(termr + termsum + termy)
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
    return {'acc y1': num_y1/len(p1), 'acc y2': num_y2/len(p2), 'good reject': num_good_rs/len(p1), 'bad reject': num_bad_rs/len(p1), 'good non-reject': num_good_nors/len(p1), 'bad non-reject': num_bad_nors/len(p1)}
# model = model()

def exp(d=32*20, cost=0.01):


    # d = 32*20
    # cost=0.01
    # cost = cost

    c = torch.ones(d)*cost

    # x = torch.normal(mean=torch.zeros((1, d)), std=torch.ones((1, d)))
    # z = torch.normal(mean=torch.zeros((1, d)), std=torch.ones((1, d)))

    x = torch.normal(mean=torch.zeros((d,1)), std=torch.ones((d,1)))
    z = torch.normal(mean=torch.zeros((d,1)), std=torch.ones((d,1)))

    # y = (torch.ones((d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x*z))).type(torch.LongTensor)
    y = torch.ones((d, 2))
    for i in range(len(x)):
        y[int(torch.bernoulli(torch.nn.Sigmoid()(x[i]*z[i])))] = 1
        y[int(1-torch.bernoulli(torch.nn.Sigmoid()(x[i]*z[i])))] = 0
    # y = torch.ones((d, 1))

    x_val = torch.normal(mean=torch.zeros((d,1)), std=torch.ones((d,1)))
    z_val = torch.normal(mean=torch.zeros((d,1)), std=torch.ones((d,1)))

    # y_val = torch.ones((d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x*z))
    # y_val = torch.ones((d, 1))

    y_val = torch.ones((d, 2))
    for i in range(len(x)):
        y_val[int(torch.bernoulli(torch.nn.Sigmoid()(x_val[i]*z_val[i])))] = 1
        y_val[int(1-torch.bernoulli(torch.nn.Sigmoid()(x_val[i]*z_val[i])))] = 0

    x_test = torch.normal(mean=torch.zeros((d*100,1)), std=torch.ones((d*100,1)))
    z_test = torch.normal(mean=torch.zeros((d*100,1)), std=torch.ones((d*100,1)))

    # y_test = torch.ones((d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x*z))
    # y_test = torch.ones((d, 1))
    y_test = torch.ones((d, 2))
    for i in range(len(x)):
        y_test[int(torch.bernoulli(torch.nn.Sigmoid()(x_test[i]*z_test[i])).item())] = 1
        y_test[int(1-torch.bernoulli(torch.nn.Sigmoid()(x_test[i]*z_test[i])).item())] = 0


    y1 = mlp(feat_dim=1, out_dim=3, hidden_dim=100)
    y2 = mlp(feat_dim=2, out_dim=2, hidden_dim=200)

    jm = model(feat_dim=1, out_dim=2, hidden_dim=100)
    # s = mlp(feat_dim=3, hidden_dim=200)


    epoch=1
    batch_size=32
    optimizer = torch.optim.Adam(jm.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # breakpoint()
    # start_params = list(y1.parameters().clone()) + list(y2.parameters().clone())
    running_loss = 0
    for i in range(epoch):
        for batch in range(batch_size, d+batch_size, batch_size):

            x_batch = x[batch-batch_size:batch]
            z_batch = z[batch-batch_size:batch]
            y_batch = y[batch-batch_size:batch]

            optimizer.zero_grad()


            # loss = loss_fn(x_batch, z_batch, y_batch, cost, y1, y2)
            loss = loss_fn(x_batch, z_batch, y_batch, cost, jm, None)

            loss.backward()
            # for i,p in enumerate(jm.parameters()):
            #     print(i, p.grad.norm())
            optimizer.step()

            running_loss += loss.item()
            if batch % 10 == 0:
                # last_loss = running_loss / 320 # loss per batch
                print('  batch {} loss: {}'.format(batch/batch_size, loss))
              
                valid = eval(x_val[batch-batch_size:batch], z_val[batch-batch_size:batch], y_val[batch-batch_size:batch], cost, jm,None)
                print(valid)
            scheduler.step()

    result = {}
    for key in valid.keys():
        result[key] = []
    for batch in range(batch_size, d+batch_size, batch_size):

        x_batch = x_test[batch-batch_size:batch]
        z_batch = z_test[batch-batch_size:batch]
        y_batch = y_test[batch-batch_size:batch]


        out = eval(x_batch, z_batch, y_batch, cost, jm, None, debug=True)
        for key, value in out.items():
            result[key].append(value)

    mean_result = {}
    sdd_result = {}
    for key, value in result.items():
        mean_result[key] = np.mean(value)
        sdd_result[key] = np.std(value)
    # print(start_params==list(y1.parameters()) + list(y2.parameters()))
    # breakpoint()
    return mean_result, sdd_result

costs = [0.001, 0.002, 0.005, 0.1, 0.2, 0.5, 1, 2, 5, 10]
# costs = list(np.arange(0, 1, 0.2))
# costs = np.arange()
# costs = [10000000]
# costs = [0.0000001]
# costs = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
# costs = [0, 0, 0,0,0]
# costs = [10000, 10000, 10000,10000]
costs = np.arange(0, 1, 0.01)
num_accept = []
num_reject = []
accept_acc = []
reject_acc = []
y1_acc = []
y2_acc = []
l_sdd = []

for cost in costs:
    result, sdd = exp(d=32*1000,cost=cost)
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
