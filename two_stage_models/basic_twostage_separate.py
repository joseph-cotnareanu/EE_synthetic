import torch
import torch.nn as nn

class NNTwoStageSeparate(torch.nn.Module):
    def __init__(self,x_dim:int, z_dim:int, num_classe:int):
        
        super(NNTwoStageSeparate, self).__init__()
        
        self.param_tracking_dict = {}
        self.y1_in = nn.Linear(1,4)
        torch.nn.init.xavier_uniform(self.y1_in.weight)
        self.y1_out = nn.Linear(4,1)
        torch.nn.init.xavier_uniform(self.y1_out.weight)
        # self.y2_in = nn.Linear(2,4)
        self.y2_in = nn.Linear(1,4)
        torch.nn.init.xavier_uniform(self.y2_in.weight)

        self.y2_out = nn.Linear(4,1)
        torch.nn.init.xavier_uniform(self.y2_out.weight)
        self.s_in = nn.Linear(1,4)
        torch.nn.init.xavier_uniform(self.s_in.weight)
        self.s_out = nn.Linear(4,1)
        torch.nn.init.xavier_uniform(self.s_out.weight)
        self.tanh = nn.Tanh()
        
        self.sigmoid = torch.nn.Sigmoid()
       
        self.relu = torch.nn.ReLU()
 
        
    def forward(self, x,z, debug):
        # y1 = x*self.a_y1  + self.b_y1
        # y2 =(x+z)* self.c_y2  + self.d_y2
        
        # s = self.sigmoid(x-self.s_1)*self.sigmoid(self.s_2-x)
        # # s=1.0
        # # if debug: breakpoint()        
        # # print(self.c_y2)
        # breakpoint()
        y1 = self.tanh(self.y1_out(self.tanh(self.y1_in(x))))
        # y2 = self.tanh(self.y2_out(self.tanh(self.y2_in(torch.cat((x,z), dim=-1)))))
        y2 = self.tanh(self.y2_out(self.tanh(self.y2_in(x+z))))

        s = self.sigmoid(self.s_out(self.tanh(self.s_in(x))))
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
