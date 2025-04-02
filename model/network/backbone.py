import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import copy
from .basic_blocks import Clones, ParseNorm, BINorm
from .conv_blocks import BasicConv2D, ResConv2D, BasicConv2DBlock, ResConv2DBlock

class PlainBackbone(nn.Module):
    def __init__(self, channels, in_channel, in_height, in_width, norm_bin, norm_type, norm_affine, ConvBlock=BasicConv2DBlock):
        super(PlainBackbone, self).__init__()
        # params
        self.channels = channels
        self.in_channel = in_channel
        self.in_height = in_height
        self.in_width = in_width
        self.norm_bin = norm_bin
        self.norm_type = norm_type
        self.norm_affine = norm_affine
        # layers
        if len(self.channels) > 3:
            self.backbone = nn.Sequential(
                BasicConv2D(in_channel, channels[0], in_height, in_width, norm_bin, norm_type, norm_affine, kernel_size=5, stride=1, padding=2),
                BasicConv2D(channels[0], channels[0], in_height, in_width, norm_bin, norm_type, norm_affine, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2),
                ConvBlock(channels[0], channels[1], in_height//2, in_width//2, norm_bin, norm_type, norm_affine, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2),
                ConvBlock(channels[1], channels[2], in_height//4, in_width//4, norm_bin, norm_type, norm_affine, kernel_size=3, stride=1, padding=1),
                ConvBlock(channels[2], channels[3], in_height//4, in_width//4, norm_bin, norm_type, norm_affine, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.backbone = nn.Sequential(
                BasicConv2D(in_channel, channels[0], in_height, in_width, norm_bin, norm_type, norm_affine, kernel_size=5, stride=1, padding=2),
                BasicConv2D(channels[0], channels[0], in_height, in_width, norm_bin, norm_type, norm_affine, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2),
                ConvBlock(channels[0], channels[1], in_height//2, in_width//2, norm_bin, norm_type, norm_affine, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2),
                ConvBlock(channels[1], channels[2], in_height//4, in_width//4, norm_bin, norm_type, norm_affine, kernel_size=3, stride=1, padding=1)
            )

    def extra_repr(self):
        return 'channels={}, in_channel={}, in_height={}, in_width={}, norm_bin={}, norm_type={}, norm_affine={}'.format( \
                    self.channels, self.in_channel, self.in_height, self.in_width, self.norm_bin, self.norm_type, self.norm_affine) 

    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.backbone(x.view(-1, c, h, w))
        _, c, h, w = x.size()
        x = x.view(n, s, c, h, w)
        return x

class ResconvBackbone(nn.Module):
    def __init__(self, channels, in_channel, in_height, in_width, norm_bin, norm_type, norm_affine):
        super(ResconvBackbone, self).__init__()
        # params
        self.channels = channels
        self.in_channel = in_channel
        self.in_height = in_height
        self.in_width = in_width
        self.norm_bin = norm_bin
        self.norm_type = norm_type
        self.norm_affine = norm_affine
        # layers
        if len(self.channels) > 3:
            self.backbone = nn.Sequential(
                BasicConv2D(in_channel, channels[0], in_height, in_width, norm_bin, norm_type, norm_affine, kernel_size=5, stride=1, padding=2),
                ResConv2DBlock(channels[0], channels[0], in_height, in_width, norm_bin, norm_type, norm_affine, kernel_size=3, stride=1, padding=1),
                ResConv2DBlock(channels[0], channels[1], in_height//2, in_width//2, norm_bin, norm_type, norm_affine, kernel_size=3, stride=2, padding=1),
                ResConv2DBlock(channels[1], channels[2], in_height//4, in_width//4, norm_bin, norm_type, norm_affine, kernel_size=3, stride=2, padding=1),
                ResConv2DBlock(channels[2], channels[3], in_height//4, in_width//4, norm_bin, norm_type, norm_affine, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.backbone = nn.Sequential(
                BasicConv2D(in_channel, channels[0], in_height, in_width, norm_bin, norm_type, norm_affine, kernel_size=5, stride=1, padding=2),
                ResConv2DBlock(channels[0], channels[0], in_height, in_width, norm_bin, norm_type, norm_affine, kernel_size=3, stride=1, padding=1),
                ResConv2DBlock(channels[0], channels[1], in_height//2, in_width//2, norm_bin, norm_type, norm_affine, kernel_size=3, stride=2, padding=1),
                ResConv2DBlock(channels[1], channels[2], in_height//4, in_width//4, norm_bin, norm_type, norm_affine, kernel_size=3, stride=2, padding=1)
            )

    def extra_repr(self):
        return 'channels={}, in_channel={}, in_height={}, in_width={}, norm_bin={}, norm_type={}, norm_affine={}'.format( \
                    self.channels, self.in_channel, self.in_height, self.in_width, self.norm_bin, self.norm_type, self.norm_affine) 

    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.backbone(x.view(-1, c, h, w))
        _, c, h, w = x.size()
        x = x.view(n, s, c, h, w)
        return x