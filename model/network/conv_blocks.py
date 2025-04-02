import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import copy
from .basic_blocks import Clones, ParseNorm, BINorm

class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, out_height, out_width, norm_bin, norm_type, norm_affine, kernel_size, stride, **kwargs):
        super(BasicConv2D, self).__init__()
        assert(norm_type in ['None', 'BN', 'Sync_BN', 'DDP_Sync_BN', 'IN', 'LN'])
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False, **kwargs), 
            BINorm(norm_bin, norm_type, norm_affine, out_channels, out_height, out_width),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.layers(x)

class ResConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, out_height, out_width, norm_bin, norm_type, norm_affine, kernel_size, stride, **kwargs):
        super(ResConv2D, self).__init__()
        assert(norm_type in ['None', 'BN', 'Sync_BN', 'DDP_Sync_BN', 'IN', 'LN'])
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False, **kwargs),
            BINorm(norm_bin, norm_type, norm_affine, out_channels, out_height, out_width)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BINorm(norm_bin, norm_type, norm_affine, out_channels, out_height, out_width)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BasicConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_height, out_width, norm_bin, norm_type, norm_affine, kernel_size, stride, **kwargs):
        super(BasicConv2DBlock, self).__init__()
        assert(norm_type in ['None', 'BN', 'Sync_BN', 'DDP_Sync_BN', 'IN', 'LN'])
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False, **kwargs),
            BINorm(norm_bin, norm_type, norm_affine, out_channels, out_height, out_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, bias=False, **kwargs),
            BINorm(norm_bin, norm_type, norm_affine, out_channels, out_height, out_width),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.layers(x)

class ResConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_height, out_width, norm_bin, norm_type, norm_affine, kernel_size, stride, **kwargs):
        super(ResConv2DBlock, self).__init__()
        assert(norm_type in ['None', 'BN', 'Sync_BN', 'DDP_Sync_BN', 'IN', 'LN'])
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False, **kwargs),
            BINorm(norm_bin, norm_type, norm_affine, out_channels, out_height, out_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, bias=False, **kwargs),
            BINorm(norm_bin, norm_type, norm_affine, out_channels, out_height, out_width)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BINorm(norm_bin, norm_type, norm_affine, out_channels, out_height, out_width)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        