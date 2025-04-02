import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class NormalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    def forward(self, input):
        out = F.linear(input, self.weight).float()
        return out

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, scale='fixed_16'):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_type, self.sigma_init = scale.split('_')
        assert(self.sigma_type in ['fixed', 'unfixed'])
        if self.sigma_type == 'fixed':
            self.sigma = float(self.sigma_init)
        elif self.sigma_type == 'unfixed':
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(float(self.sigma_init))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # if self.sigma is not None:
        #     self.sigma.data.fill_(float(self.sigma_init)) #for initializaiton of sigma
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, scale={}'.format(self.in_features, self.out_features, self.scale)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1)).float()
        out = self.sigma * out
        return out