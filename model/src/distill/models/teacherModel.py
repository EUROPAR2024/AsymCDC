from torch import nn
import torch
import time

class TeacherModel(nn.Module):
    def __init__(self, model, ec_k, dataset=''):
        super(TeacherModel, self).__init__()
        self.model = model
        self.ec_k = ec_k
        self.dataset = dataset
    
    def forward(self, y):
        assert(len(y) % self.ec_k == 0)
        # compute x based on y
        # start_t1 = time.time()
        if self.dataset == 'fashion':
            x = self.model.inverse(y)
        else:
            x = self.model.module.inverse(y)
        # end_t1 = time.time()
        # print("inverse time costs: {} ms".format(float(end_t1 - start_t1)*1000.0))
        parity_num = len(y) // self.ec_k
        out = []
        for i in range(parity_num):
            xp = x[(i+1)*self.ec_k-1]
            
            # compute failed xj based on x
            for j in range(self.ec_k-1):
                xp -= (1 / self.ec_k) * x[i*self.ec_k+j]
                
            xp *= self.ec_k
            out.append(xp)
        out = torch.stack(out)
        # start_t2 = time.time()
        zj, yj = self.model(out)
        # end_t2 = time.time()
        # print("forward time costs: {} ms".format(end_t2 - start_t2))
        return zj