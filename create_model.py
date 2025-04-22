import torch

import torch
import torch.nn as nn

class NNTwoStageSeparate(torch.nn.Module):
    def __init__(self,x_dim:int, z_dim:int, num_classe:int):
        
        super(NNTwoStageSeparate, self).__init__()
        hidden_dim = 128
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

class BasicTwoStageSeparate(torch.nn.Module):
    def __init__(self,x_dim:int, z_dim:int, num_classe:int):
        """Simple 6 parameter model, with a separate decision module:
        y1 = sigmoid(ax+b)
        y1 = sigmoid(cx+d)
        s = sigmoid(x-e)sigmoid(f-x)
           

        Args:
            x_dim int: 
            z_dim int: 
            num_classe int: 
        """
        super(BasicTwoStageSeparate, self).__init__()
        
        self.param_tracking_dict = {}
        
        self.a_y1 = nn.Parameter(torch.zeros((1,1), dtype=float, requires_grad=True))
        self.b_y1 = nn.Parameter(torch.zeros((1,1), dtype=float, requires_grad=True))
        self.c_y2 = nn.Parameter(torch.zeros((1,1), dtype=float, requires_grad=True))
        self.d_y2 = nn.Parameter(torch.zeros((1,1), dtype=float, requires_grad=True))
        
        self.s_1 = nn.Parameter(torch.zeros((1), dtype=float, requires_grad=True))
        self.s_2 = nn.Parameter(torch.zeros((1), dtype=float, requires_grad=True))
        
        self.params = nn.ParameterList([self.a_y1, self.b_y1,self.c_y2,self.d_y2, self.s_1, self.s_2])
        
        self.sigmoid = torch.nn.Sigmoid()
       
        
    def forward(self, x,z, debug):
        y1 = x*self.a_y1  + self.b_y1
        y2 =(x+z)* self.c_y2  + self.d_y2
        
        s = self.sigmoid(x-self.s_1)*self.sigmoid(self.s_2-x)
       
        param_tracking_dict = {'a' : self.a_y1, 'b':self.b_y1, 'c':self.c_y2, 'd':self.d_y2, 's':s}
        return y1, y2, s, param_tracking_dict

def create_two_stage_model(x_dim:int, z_dim:int, num_classes:int, two_stage_model_name):
    if two_stage_model_name == 'linear':
        two_stage_model = BasicTwoStageSeparate(x_dim, z_dim, num_classes)
    elif two_stage_model_name == 'NN':
        two_stage_model = NNTwoStageSeparate(x_dim, z_dim, num_classes)
    # torch.nn.init.xavier_uniform(two_stage_model.weight)
    return two_stage_model
