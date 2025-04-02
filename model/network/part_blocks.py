import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import copy
from .basic_blocks import Clones, ParseNorm

class PartBlock_OneFC(nn.Module):
    def __init__(self, num_parts, in_channels, out_channels, \
                    pre_norm_type='None', pre_norm_affine=False, mid_norm_type='None', mid_norm_affine=False, after_norm_type='LN', after_norm_affine=True):
        super(PartBlock_OneFC, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pre_norm_type = pre_norm_type
        self.pre_norm_affine = pre_norm_affine
        self.mid_norm_type = mid_norm_type
        self.mid_norm_affine = mid_norm_affine
        self.after_norm_type = after_norm_type
        self.after_norm_affine = after_norm_affine
        part_block = nn.Sequential(
                            ParseNorm(pre_norm_type, pre_norm_affine, in_channels, height=None, width=None),
                            nn.Linear(in_channels, out_channels, bias=False),
                            ParseNorm(mid_norm_type[0], mid_norm_affine[0], out_channels, height=None, width=None),
                            )
        self.part_block = Clones(part_block, self.num_parts)
        self.part_norm = ParseNorm(after_norm_type, after_norm_affine, out_channels, height=None, width=None)

    def extra_repr(self):
        return 'num_parts={}, in_channels={}, out_channels={}, pre_norm_type={}, pre_norm_affine={}, mid_norm_type={}, mid_norm_affine={}, after_norm_type={}, after_norm_affine={}'.format( \
                    self.num_parts, self.in_channels, self.out_channels, self.pre_norm_type, self.pre_norm_affine, self.mid_norm_type, self.mid_norm_affine, self.after_norm_type, self.after_norm_affine) 

    def forward_norm(self, x):
        m, n, d = x.size()
        x = self.part_norm(x.view(-1, d)).view(m, n, d)
        return x

    def forward(self, x):
        # input: num_parts x batch_size x in_channels, output: num_parts x batch_size x out_channels
        out = list()
        for part_x, part_block in zip(x.split(1, 0), self.part_block):
            out.append(part_block(part_x.squeeze(0)).unsqueeze(0))
        out = torch.cat(out, 0)
        out = self.forward_norm(out)    
        return out