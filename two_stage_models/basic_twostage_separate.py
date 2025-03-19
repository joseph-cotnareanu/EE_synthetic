import torch
import torch.nn as nn

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
        
       
            
        self.a_y1 = nn.Parameter(torch.zeros((1,1), dtype=float, requires_grad=True))
        self.b_y1 = nn.Parameter(torch.zeros((1,1), dtype=float, requires_grad=True))
        self.c_y2 = nn.Parameter(torch.zeros((1,1), dtype=float, requires_grad=True))
        self.d_y2 = nn.Parameter(torch.zeros((1,1), dtype=float, requires_grad=True))
        
        self.s_1 = nn.Parameter(torch.zeros((1), dtype=float, requires_grad=True))
        self.s_2 = nn.Parameter(torch.zeros((1), dtype=float, requires_grad=True))
        
        self.params = nn.ParameterList([self.a_y1, self.b_y1,self.c_y2,self.d_y2, self.s_1, self.s_2])
        
        self.sigmoid = torch.nn.Sigmoid()
        
 
        
    def forward(self, x,z, debug):
        y1 = x*self.a_y1 + self.b_y1
        y2 = (x+z)* self.c_y2  + self.d_y2
        
        s = self.sigmoid(x-self.s_1)*self.sigmoid(self.s_2-x)
        s=1
        # if debug: breakpoint()        
        # print(self.c_y2)
        return y1, y2, s, self.c_y2, self.d_y2
