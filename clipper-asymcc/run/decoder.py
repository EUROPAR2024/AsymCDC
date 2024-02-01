import torch
import torch.nn as nn
from util import try_cuda, get_flattened_dim
import numpy as np

class Decoder():
    def __init__(self, ec_k) -> None:
        self.ec_k = ec_k

    def __call__(self):
        pass
    
class LinearDecoder(Decoder):
    def __call__(self, input):
        out, outbij = input
        return out[-1] * self.ec_k - torch.sum(out[:-1], dim=0)
    
class DistilledDecoder(Decoder):
    def __init__(self, ec_k, model) -> None:
        super().__init__(ec_k)
        self.model = model
        
    def __call__(self, input):
        out, outbij = input
        outbij = outbij.unsqueeze(0)
        if torch.cuda.is_available():
            outbij = outbij.cuda()
        y, _ = self.model(outbij)
        return y.cpu().data[0]

class MLPDecoder(Decoder):
    def __init__(self, ec_k, in_dim):
        super().__init__(ec_k)
        self.inout_dim = in_dim[0] # 10
        num_in = self.ec_k + 1
        num_out = self.ec_k
        
        self.nn = nn.Sequential(
            nn.Linear(in_features=num_in * self.inout_dim,
                      out_features=num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=num_in * self.inout_dim,
                      out_features=num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=num_in * self.inout_dim,
                      out_features=2*num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=2*num_in * self.inout_dim,
                      out_features=2*num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=2*num_in * self.inout_dim,
                      out_features=4*num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=4*num_in * self.inout_dim,
                      out_features=4*num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=4*num_in * self.inout_dim,
                      out_features=4*num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=4*num_in * self.inout_dim,
                      out_features=2*num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=2*num_in * self.inout_dim,
                      out_features=num_in * self.inout_dim),
            nn.ReLU(),
            nn.Linear(in_features=num_in * self.inout_dim,
                      out_features=num_out * self.inout_dim)
        )

    def __call__(self, input): 
        out, outbij = input
        fail = -1
        for i,data in enumerate(out):
            if np.count_nonzero(data) == 0:
                fail = i
        val = out.view(-1) #[(k+1), 10] --> [(k+1)*10]
        y = self.nn(val)  # --> [k*10]
        y = y.view(self.ec_k, -1) # --> [k, 10]
        return y.data[fail]
