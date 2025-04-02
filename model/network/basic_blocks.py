import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import copy
from .sync_batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d 

def Clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class BINorm(nn.Module):
    def __init__(self, norm_bin, norm_type, norm_affine, channels, height, width, threed=False):
        super(BINorm, self).__init__()
        self.norm_bin = norm_bin
        self.norm_type = norm_type
        self.norm_affine = norm_affine
        self.channels = channels
        self.height = height
        self.width = width
        self.threed = threed
        self.norm_layer = Clones(ParseNorm(norm_type, norm_affine, channels, int(height//norm_bin), width), norm_bin)

    def extra_repr(self):
        return 'norm_bin={}, norm_type={}, norm_affine={}, channels={}, height={}, width={}, threed={}'.format( \
                    self.norm_bin, self.norm_type, self.norm_affine, self.channels, self.height, self.width, self.threed)

    def layer_norm_3d(self, layer, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous() # N x D x C x H x W
        x = layer(x) # N x D x C x H x W
        x = x.permute(0, 2, 1, 3, 4).contiguous() # N x C x D x H x W
        return x

    def forward(self, x):
        if self.threed:
            N, C, D, H, W = x.size()
            assert(H % self.norm_bin == 0)
            x = x.view(N, C, D, self.norm_bin, int(H/self.norm_bin), W)
            if self.norm_type == 'LN':
                x = torch.cat([(self.layer_norm_3d(_norm, _x.squeeze(3))).unsqueeze(3) for _x, _norm in zip(x.split(1, dim=3), self.norm_layer)], dim=3)
            else:
                x = torch.cat([_norm(_x.squeeze(3)).unsqueeze(3) for _x, _norm in zip(x.split(1, dim=3), self.norm_layer)], dim=3)
            x = x.view(N, C, D, H, W)
        else:
            N, C, H, W = x.size()
            assert(H % self.norm_bin == 0)
            x = x.view(N, C, self.norm_bin, int(H/self.norm_bin), W)
            x = torch.cat([_norm(_x.squeeze(2)).unsqueeze(2) for _x, _norm in zip(x.split(1, dim=2), self.norm_layer)], dim=2)
            x = x.view(N, C, H, W)
        return x

def ParseNorm(norm_type, norm_affine, channels, height, width):
    assert(norm_type in ['None', 'BN', 'Sync_BN', 'DDP_Sync_BN', 'IN', 'LN'] or \
            norm_type in ['BN3D', 'Sync_BN3D', 'DDP_Sync_BN3D', 'IN3D'])
    if norm_type == 'None':
        return nn.Identity()
    elif norm_type == 'BN':
        return nn.BatchNorm2d(channels, affine=norm_affine)
    elif norm_type == 'BN3D':
        return nn.BatchNorm3d(channels, affine=norm_affine)
    elif norm_type == 'Sync_BN':
        return SynchronizedBatchNorm2d(channels, affine=norm_affine)
    elif norm_type == 'Sync_BN3D':
        return SynchronizedBatchNorm3d(channels, affine=norm_affine)
    elif norm_type == 'DDP_Sync_BN':
        return nn.SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm2d(channels, affine=norm_affine))
    elif norm_type == 'DDP_Sync_BN3D':
        return nn.SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm3d(channels, affine=norm_affine))
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(channels, affine=norm_affine)
    elif norm_type == 'IN3D':
        return nn.InstanceNorm3d(channels, affine=norm_affine)
    elif norm_type == 'LN':
        if height is None or width is None:
            return nn.LayerNorm(channels, elementwise_affine=norm_affine) # for fully-connected layers
        else:
            return nn.LayerNorm([channels, height, width], elementwise_affine=norm_affine) # for convolutional layers
