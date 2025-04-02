import os
import os.path as osp
import numpy as np
import math
import cv2
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

class OverlapLoss(nn.Module):
    def __init__(self):
        super(OverlapLoss, self).__init__()  

    def forward(self, attack_seq, refer_seq, edge_mask=None):
        '''
        attack_seq: 1 x num_template x height x width
        refer_seq: 1 x num_template x height x width
        edge_mask: 1 x num_template x height x width
        '''
        loss = (attack_seq - refer_seq) ** 2 # 1 x num_template x height x width
        if edge_mask is None:
            edge_loss = torch.mean(loss) # 1
        else:
            edge_loss = torch.sum(loss * edge_mask, dim=[2, 3]) / torch.sum(edge_mask, dim=[2, 3]) # 1 x num_template
            edge_loss = torch.mean(edge_loss) # 1
            # # DEBUG
            # non_edge_loss = torch.sum(loss * (1-edge_mask), dim=[2, 3]) / torch.sum((1-edge_mask), dim=[2, 3]) # 1 x num_template
            # non_edge_loss = torch.mean(non_edge_loss) # 1
            # print('non_edge_loss={}'.format(non_edge_loss))
        return edge_loss

class PushAwayLoss(nn.Module):
    def __init__(self, margin=10.0):
        super(PushAwayLoss, self).__init__()
        self.margin = margin
    
    def forward(self, attack_feat, refer_feat):
        '''
        attack_feat: 1 x num_parts x feat_dim
        refer_feat: 1 x num_parts x feat_dim
        '''
        dist = (attack_feat - refer_feat) ** 2 # 1 x num_parts x feat_dim
        dist = torch.sum(dist, dim=2) # 1 x num_parts
        dist = torch.sqrt(F.relu(dist)) # 1 x num_parts
        mean_dist = torch.mean(dist, dim=1) # 1
        loss = F.relu(self.margin - mean_dist) # 1
        # print('dist={}, mean_dist={}, loss={}'.format(dist, mean_dist, loss))
        return loss

class PullCloseLoss(nn.Module):
    def __init__(self, agg_type='MEAN'):
        super(PullCloseLoss, self).__init__()
        self.agg_type = agg_type
        assert(self.agg_type in ['MeanFeat', 'MeanDist', 'MaxDist'])
    
    def forward(self, attack_feat, fake_feat):
        '''
        attack_feat: 1 x num_parts x feat_dim
        fake_feat: num_fake_gallery x num_parts x feat_dim
        '''
        if self.agg_type == 'MeatFeat':
            fake_feat = torch.mean(fake_feat, dim=0, keepdim=True)
        n = fake_feat.size(0)
        attack_feat = attack_feat.repeat(n, 1, 1) # n x num_parts x feat_dim
        dist = (attack_feat - fake_feat) ** 2 # n x num_parts x feat_dim
        dist = torch.sum(dist, dim=2) # n x num_parts
        dist = torch.sqrt(F.relu(dist)) # n x num_parts
        mean_dist = torch.mean(dist, dim=1) # n
        if self.agg_type == 'MaxDist':
            loss = torch.max(mean_dist)[0]
        else:
            loss = torch.mean(mean_dist)
        # print('dist={}, mean_dist={}, loss={}'.format(dist, mean_dist, loss))
        return loss