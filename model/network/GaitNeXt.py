import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import numpy as np
from copy import deepcopy
from .sync_batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from .basic_blocks import Clones, ParseNorm, BINorm
from .conv_blocks import BasicConv2D, ResConv2D, BasicConv2DBlock, ResConv2DBlock
from .backbone import PlainBackbone, ResconvBackbone
from .part_blocks import PartBlock_OneFC
from .linear_blocks import NormalLinear, CosineLinear

class GaitNeXt(nn.Module):
    def __init__(self, config):
        super(GaitNeXt, self).__init__()
        self.config = deepcopy(config)
        #############################################################
        # encoder backbone
        in_channel = 1
        in_height = self.config['resolution']
        in_width = self.config['resolution'] - 2*int(float(self.config['resolution'])/64*10)
        if self.config['backbone'] == 'Plain':
            self.encoder_backbone = PlainBackbone(self.config['channels'], in_channel, in_height, in_width, \
                                        self.config['base_norm_bin'], self.config['base_norm_type'], self.config['base_norm_affine'], ConvBlock=BasicConv2DBlock)
        elif self.config['backbone'] == 'Respool':
            self.encoder_backbone = PlainBackbone(self.config['channels'], in_channel, in_height, in_width, \
                                        self.config['base_norm_bin'], self.config['base_norm_type'], self.config['base_norm_affine'], ConvBlock=ResConv2DBlock)
        elif self.config['backbone'] == 'Resconv':
            self.encoder_backbone = ResconvBackbone(self.config['channels'], in_channel, in_height, in_width, \
                                        self.config['base_norm_bin'], self.config['base_norm_type'], self.config['base_norm_affine'])
        #############################################################
        # part align
        frame_att_output_dim = self.config['channels'][-1]
        #############################################################
        # part block
        num_parts = sum(list(self.config['bin_num']))
        if self.config['pmap_type'] == 'OneFC':
            self.fc_bin = PartBlock_OneFC(num_parts, frame_att_output_dim, self.config['hidden_dim'], \
                            pre_norm_type=self.config['pmap_pre_norm_type'], pre_norm_affine=self.config['pmap_pre_norm_affine'], \
                            mid_norm_type=self.config['pmap_mid_norm_type'], mid_norm_affine=self.config['pmap_mid_norm_affine'], \
                            after_norm_type=self.config['pmap_after_norm_type'], after_norm_affine=self.config['pmap_after_norm_affine'])
        #############################################################
        # BNNeck
        if self.config['encoder_entropy_weight'] > 0 or self.config['encoder_supcon_weight'] > 0:
            if self.config['separate_bnneck']:
                if self.config['DDP']:
                    part_bn = nn.BatchNorm1d(self.config['hidden_dim'])
                    part_bn = nn.SyncBatchNorm.convert_sync_batchnorm(part_bn)
                else:
                    part_bn = SynchronizedBatchNorm1d(self.config['hidden_dim'])
                self.part_bn = Clones(part_bn, num_parts)
                if self.training and self.config['encoder_entropy_weight'] > 0:
                    self.part_drop = nn.Dropout(self.config['encoder_entropy_linear_drop'], inplace=True)
                    # part_cls = nn.Linear(self.config['hidden_dim'], self.config['num_id'], bias=False)
                    if self.config['encoder_entropy_linear_type'] == 'NORMAL':
                        part_cls = NormalLinear(self.config['hidden_dim'], self.config['num_id'])
                    elif self.config['encoder_entropy_linear_type'] == 'COSINE':
                        part_cls = CosineLinear(self.config['hidden_dim'], self.config['num_id'], self.config['encoder_entropy_linear_scale'])
                    self.part_cls = Clones(part_cls, num_parts)
                # Initialization
                for _layer in self.part_bn:
                    _layer.bias.requires_grad_(False)
                if self.training and self.config['encoder_entropy_weight'] > 0:
                    for _layer in self.part_cls:
                        nn.init.normal_(_layer.weight, 0, 0.001)
            else:
                if self.config['DDP']:
                    self.part_bn = nn.BatchNorm1d(self.config['hidden_dim']*num_parts)
                    self.part_bn = nn.SyncBatchNorm.convert_sync_batchnorm(self.part_bn)
                else:
                    self.part_bn = SynchronizedBatchNorm1d(self.config['hidden_dim']*num_parts)
                if self.training and self.config['encoder_entropy_weight'] > 0:
                    self.part_drop = nn.Dropout(self.config['encoder_entropy_linear_drop'], inplace=True)
                    # self.part_cls = nn.Linear(self.config['hidden_dim']*num_parts, self.config['num_id'], bias=False)
                    if self.config['encoder_entropy_linear_type'] == 'NORMAL':
                        self.part_cls = NormalLinear(self.config['hidden_dim']*num_parts, self.config['num_id'])
                    elif self.config['encoder_entropy_linear_type'] == 'COSINE':
                        self.part_cls = CosineLinear(self.config['hidden_dim']*num_parts, self.config['num_id'], self.config['encoder_entropy_linear_scale'])
                # Initialization
                self.part_bn.bias.requires_grad_(False)
                if self.training and self.config['encoder_entropy_weight'] > 0:
                    nn.init.normal_(self.part_cls.weight, 0, 0.001)
        #############################################################         
        #initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # print('Conv Initialization: ', m)
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                # print('Linear Initialization: ', m)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, \
                        SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d, \
                        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, \
                        nn.LayerNorm, nn.SyncBatchNorm)):
                # print('Norm Initialization: ', m)
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def set_pool(self, x, batch_frames=None):
        if batch_frames is None:
            # x: N x S x C x H x W
            return torch.max(x, 1)[0] # N x C x H x W
        else:
            assert(x.size(0) == 1) # 1 x S x C x H x W
            x = x.squeeze(0) # S x C x H x W
            tmp = []
            for i in range(len(batch_frames) - 1):
                tmp.append(torch.max(x[batch_frames[i]:batch_frames[i+1], :, :, :], 0, keepdim=True)[0])
            return torch.cat(tmp, 0) # N x C x H x W

    def forward(self, silho, batch_frames=None, label=None):
        with autocast(enabled=self.config['AMP']):
            #############################################################
            # random input: batch_size x frame_num x height x width
            # random_fn input: 1 x total_frame_num x height x width 
            if batch_frames is not None:
                batch_frames = batch_frames[0].data.cpu().numpy().tolist()
                num_seqs = len(batch_frames)
                for i in range(len(batch_frames)):
                    if batch_frames[-(i + 1)] > 0:
                        break
                    else:
                        num_seqs -= 1
                batch_frames = batch_frames[:num_seqs]
                frame_sum = np.sum(batch_frames)
                if frame_sum < silho.size(1):
                    silho = silho[:, :frame_sum, :, :]
                batch_frames = [0] + np.cumsum(batch_frames).tolist()
            x = silho.unsqueeze(2)
            del silho
            #############################################################
            # encoder backbone
            # 2D/3D (fixed_unordered/fixed_ordered)
            # input: batch_size x frame_num x 1 x height x width, output: batch_size x frame_num x last_channels x height x width
            # 2D (unfixed_unordered)
            # input: 1 x total_frame_num x 1 x height x weight, output: 1 x total_frame_num x last_channels x height x width
            # 2D+DA (fixed_unordered/fixed_ordered)
            # input: batch_size x frame_num x 1 x height x width, output: batch_size x last_frame_num x last_channels x height x width
            x = self.encoder_backbone(x) 
            #############################################################
            # set pool + part block
            x = self.set_pool(x, batch_frames)

            feature = list()
            offset = 0
            for num_bin in list(self.config['bin_num']):
                n2, c, h, w = x.size()
                z = x.view(n2, c, num_bin, -1).max(-1)[0] + x.view(n2, c, num_bin, -1).mean(-1)
                feature.append(z)

            feature = torch.cat(feature, dim=-1)                # n x c x num_parts
            feature = feature.permute(2, 0, 1).contiguous()     # num_parts x n x c 
            feature = self.fc_bin(feature)                      # num_parts x n x hidden_dim
            feature = feature.permute(1, 0, 2).contiguous()     # n x num_parts x hidden_dim
            #############################################################
            # BNNeck
            if self.config['encoder_entropy_weight'] > 0 or self.config['encoder_supcon_weight'] > 0:
                if self.config['separate_bnneck']:
                    bn_feature = [_layer(_input.squeeze(1)).unsqueeze(1) for _input, _layer in zip(feature.split(1, dim=1), self.part_bn)]
                    bn_feature = torch.cat(bn_feature, dim=1) # n x num_parts x hidden_dim
                    if self.training and self.config['encoder_entropy_weight'] > 0:
                        bn_feature = self.part_drop(bn_feature)
                        cls_score = [_layer(_input.squeeze(1)).unsqueeze(1) \
                                            for _input, _layer in zip(bn_feature.split(1, dim=1), self.part_cls)]
                        cls_score = torch.cat(cls_score, dim=1) # n x num_parts x num_class
                    else:
                        cls_score = torch.empty(1).cuda()
                else:
                    n2 = feature.size(0)
                    bn_feature = self.part_bn(feature.view(n2, -1)) # n x (num_parts x hidden_dim)
                    if self.training and self.config['encoder_entropy_weight'] > 0:
                        bn_feature = self.part_drop(bn_feature)
                        cls_score = self.part_cls(bn_feature) # n x num_class
                    else:
                        cls_score = torch.empty(1).cuda()
                    # unsqueeze
                    bn_feature = bn_feature.unsqueeze(1) # n x 1 x (num_parts x hidden_dim)
                    cls_score = cls_score.unsqueeze(1) # n x 1 x num_class
            else:
                bn_feature = torch.empty(1).cuda()
                cls_score = torch.empty(1).cuda()
            #############################################################
            # Encoder for DDP
            if self.config['encoder_triplet_weight'] <= 0:
                feature = feature.detach()
            if self.config['encoder_supcon_weight'] <= 0:
                bn_feature = bn_feature.detach()
            if self.config['encoder_entropy_weight'] <= 0:
                cls_score = cls_score.detach()

            return feature, bn_feature, cls_score