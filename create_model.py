import torch

import torch
import torch.nn as nn


class NNBinary(torch.nn.Module):
    def __init__(self,x_dim:int, z_dim:int, hidden_dim):
        
        super(NNBinary, self).__init__()
        # hidden_dim = 128
        self.param_tracking_dict = {}
        self.y1_in = nn.Linear(x_dim, hidden_dim)
        self.y1_hid = nn.Linear(hidden_dim, hidden_dim)
        self.y1_out = nn.Linear(hidden_dim, 1)
        self.y2_in = nn.Linear(z_dim + x_dim, hidden_dim)
        self.y2_hid = nn.Linear(hidden_dim, hidden_dim)
        self.y2_out = nn.Linear(hidden_dim, 1)
        self.s_in = nn.Linear(x_dim, hidden_dim)
        self.s_hid = nn.Linear(hidden_dim, hidden_dim)
        self.s_out = nn.Linear(hidden_dim, 1)
        
        self.sigmoid = torch.nn.Sigmoid()
       
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
 
        
    def forward(self, x,z, debug):
      
        y1 = self.tanh(self.y1_out(self.relu(self.y1_hid(self.tanh(self.y1_in(x))))))
        # y2 = self.relu(self.y2_out(self.relu(self.y2_in(torch.cat((x,z), dim=-1)))))
        y2 = self.tanh(self.y2_out(self.relu(self.y2_hid(self.relu(self.y2_in(torch.concatenate((x,z), dim=-1)))))))

        s = self.sigmoid(self.s_out(self.relu(self.s_hid(self.relu(self.s_in(x))))))
        param_tracking_dict  = {'s':s}
        return y1, y2, s, param_tracking_dict


    

class MultiClassNN(torch.nn.Module):
    def __init__(self,x_dim:int, z_dim:int, hidden_dim,nlayers, output_dim:int):
        
        super(MultiClassNN, self).__init__()


        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.y1_hid = torch.nn.Sequential()
        self.y2_hid = torch.nn.Sequential()
        self.s_hid = torch.nn.Sequential()
        for i in range(nlayers):
            self.y1_hid.append(nn.Linear(hidden_dim, hidden_dim))
            self.y1_hid.append(nn.BatchNorm1d(hidden_dim))
            self.y1_hid.append(self.relu)

            self.y2_hid.append(nn.Linear(hidden_dim, hidden_dim))
            self.y2_hid.append(nn.BatchNorm1d(hidden_dim))
            self.y2_hid.append(self.relu)

            self.s_hid.append(nn.Linear(hidden_dim, hidden_dim))
            self.s_hid.append(nn.BatchNorm1d(hidden_dim))
            self.s_hid.append(self.relu)

        self.param_tracking_dict = {}
        self.y1_in = nn.Linear(x_dim, hidden_dim)
        # self.y1_hid = nn.Linear(hidden_dim, hidden_dim)
        self.y1_out = nn.Linear(hidden_dim, output_dim-1) # the constraint is that \sum^K_k=1 f_k = 0, so we learn K-1 output, then set the last one to =-\sum^{K-1}_k=1 f_k
        self.y2_in = nn.Linear(z_dim + x_dim, hidden_dim)
        # self.y2_hid = nn.Linear(hidden_dim, hidden_dim)
        self.y2_out = nn.Linear(hidden_dim, output_dim-1)
        self.s_in = nn.Linear(x_dim, hidden_dim)
        self.s_hid = nn.Linear(hidden_dim, hidden_dim)
        self.s_out = nn.Linear(hidden_dim, 1)
        
        self.s_bn = nn.BatchNorm1d(1)
 
        
    def forward(self, x,z, debug):
      
        y1 = self.relu(self.y1_in(x))
        y2 = self.relu(self.y2_in(torch.concatenate((x,z), dim=-1)))
        s = self.relu(self.s_in(x))
        s = self.s_hid(s)
        y1 = self.y1_hid(y1)

        y2 = self.y2_hid(y2)
        y1 =self.tanh(self.y1_out(y1))

        y1 = torch.cat((y1, -y1.sum(-1)[:, None]), -1)
        y2 = self.tanh(self.y2_out(y2))

        y2 = torch.cat((y2, -y2.sum(-1)[:, None]), -1)

        s = self.s_out(s)
        s = self.s_bn(s)
        s = self.sigmoid(s)
       
        param_tracking_dict  = {'s':s}
        
        return y1, y2, s, param_tracking_dict



def create_two_stage_model(x_dim:int, z_dim:int, num_classes:int, hidden_dim, two_stage_model_name):
    
    if two_stage_model_name == 'NN':
        if num_classes == 2:
            two_stage_model = NNBinary(x_dim, z_dim, hidden_dim)
        else:
            two_stage_model = MultiClassNN(x_dim, z_dim, hidden_dim, output_dim=num_classes)
    
    return two_stage_model

def create_llm_model(x_dim:int, z_dim:int, num_classes:int, nlayers, hidden_dim, two_stage_model_name):
    if two_stage_model_name == 'NN':
        two_stage_model = MultiClassNN(x_dim, z_dim, hidden_dim, nlayers, output_dim=num_classes)
    return two_stage_model
    # torch.nn.init.xavier_uniform(two_stage_model.weight)
