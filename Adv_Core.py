

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

from Adv_Utils import *
from Adv_Loss import *

def select_attack_index(seq, sample_type, num_template):
    assert(sample_type in ['fixed_order', 'fixed_unorder', 'unfixed_order', 'unfixed_unorder'])
    num_frames, height, width = seq.shape
    # final_template
    if sample_type.split('_')[0] == 'fixed':
        assert(num_template >= 1)
        final_template = int(num_template)
    elif sample_type.split('_')[0] == 'unfixed':
        assert(num_template > 0 and num_template <= 1)
        final_template = max(1, math.ceil(num_frames * num_template))
    # attack_index
    if final_template >= num_frames:
        attack_index = np.arange(num_frames)
        final_template = num_frames
    else:
        if sample_type.split('_')[1] == 'unorder':
            attack_index = np.random.choice(np.arange(num_frames), final_template, replace=False)
        elif sample_type.split('_')[1] == 'order':
            offset = np.random.choice(np.arange(num_frames-final_template), 1, replace=False)
            attack_index = np.arange(final_template) + offset
    attack_index = np.asarray(sorted(attack_index))
    return final_template, attack_index

class TemplateAttackBlock(nn.Module):
    def __init__(self, num_template, height, width):
        super(TemplateAttackBlock, self).__init__()
        self.num_template = num_template
        self.template = nn.Parameter(torch.rand((1, num_template, height, width))) # 1 x num_template x height x width
    
    def init_template(self, refer_seq, init_thres=5):
        '''
        refer_seq: 1 x num_template x height x width
        '''
        init_template = torch.logit(refer_seq) # reverse sigmoid
        init_template = torch.clamp(init_template, min=(-1)*init_thres, max=init_thres) # clamp
        with torch.no_grad():
            self.template.copy_(init_template)
    
    def forward(self, seq, refer_seq, attack_index, edge_mask=None):
        '''
        seq: 1 x num_frames x height x width
        refer_seq: 1 x num_template x height x width
        attack_index: num_template
        edge_mask: 1 x num_tempalte x height x width
        '''
        if edge_mask is None:
            attack_seq = torch.sigmoid(self.template) # 1 x num_template x height x width
        else:
            attack_seq = torch.sigmoid(self.template) * edge_mask + refer_seq * (1 - edge_mask) # 1 x num_template x height x width
        new_seq = seq.clone() # 1 x num_frames x height x width
        new_seq[:, attack_index, :, :] = attack_seq # 1 x num_frames x height x width
        return new_seq

##########################################################################################################################

def EOAA_Attack_Gait(Encoder, src_seq, config, fake_seq=None):
    print('Attack: src_seq={}, src_min={}, src_max={}'.format(src_seq.shape, np.min(src_seq), np.max(src_seq)))
    if fake_seq is not None:
        assert(len(fake_seq) <= config['attack_num_fake_gallery'])
        print('Attack: fake_seq={}, fake_min={}, fake_max={}'.format([_.shape for _ in fake_seq], \
                    [np.min(_) for _ in fake_seq], [np.max(_) for _ in fake_seq]))
    # init model
    Encoder.eval()
    for p in Encoder.parameters():
        p.requires_grad = False

    # select attack_index and generate edge_mask
    num_template, attack_index = select_attack_index(src_seq, config['attack_sample_type'], config['attack_num_template']) # num_template
    refer_seq = src_seq[attack_index, :, :] # num_template x height x width
    if config['attack_edge_only']:
        edge_mask = list()
        for i in range(num_template):
            edge_mask.append(bidirectional_edge_mask(refer_seq[i, :, :], config['attack_edge_thres']))
            # cv2.imwrite('sil_{}.png'.format(i), (refer_seq[i, :, :]*255.0).astype('uint8'))
            # cv2.imwrite('edge_{}.png'.format(i), (edge_mask[-1]*255.0).astype('uint8'))
        edge_mask = np.stack(edge_mask, axis=0) # num_template x height x width
    else:
        edge_mask = None 
    attack_index = np2var(attack_index) # num_template
    refer_seq = np2var(refer_seq).unsqueeze(0) # 1 x num_template x height x width
    print('Attack: refer_seq={}, refer_seq_min={}, refer_seq_max={}'.format(refer_seq.size(), torch.min(refer_seq), torch.max(refer_seq)))
    if config['attack_edge_only']:
        edge_mask = np2var(edge_mask).unsqueeze(0) # 1 x num_template x height x width
        print('Attack: edge_mask={}, edge_mask_min={}, edge_mask_max={}'.format(edge_mask.size(), torch.min(edge_mask), torch.max(edge_mask)))
    
    # normal forward
    num_frames, height, width = src_seq.shape
    src_seq = np2var(src_seq).unsqueeze(0) # 1 x num_frames x height x width
    src_feat = Encoder(src_seq)[0].detach() # 1 x num_parts x feat_dim
    print('Attack: src_seq={}, src_feat={}'.format(src_seq.size(), src_feat.size()))
    if config['attack_mode'] == 'Targeted':
        fake_feat = list()
        for _seq in fake_seq:
            _seq = np2var(_seq).unsqueeze(0) # 1 x num_frames x height x width
            _feat = Encoder(_seq)[0].detach() # 1 x num_parts x feat_dim
            fake_feat.append(_feat)
            print('Attack: fake_seq={}, fake_feat={}'.format(_seq.size(), _feat.size()))
        fake_feat = torch.cat(fake_feat, dim=0) # num_fake_gallery x num_parts x feat_dim

    # optimize
    # Init Template Attack
    Attack = TemplateAttackBlock(num_template, height, width).cuda() # 1 x num_template x height x width
    if config['attack_init_template']:
        Attack.init_template(refer_seq, config['attack_init_thres'])
    # Init Loss
    if config['attack_mode'] == 'Targeted' and config['attack_pullclose_weight'] > 0:
        Attack_Pullclose_Loss = PullCloseLoss(config['attack_agg_fake_gallery'])
    if config['attack_pushaway_weight'] > 0:
        Attack_Pushaway_Loss = PushAwayLoss(config['attack_pushaway_margin'])
    if config['attack_overlap_weight'] > 0:
        Attack_Overlap_Loss = OverlapLoss()
    # Init Optimizer
    if config['attack_optimizer_type'] == 'ADAM':
        Optimizer = optim.Adam([{'params': Attack.parameters()}, {'params':Encoder.parameters()}], lr=config['attack_lr'])
    elif config['attack_optimizer_type'] == 'ADAMW':
        Optimizer = optim.AdamW([{'params': Attack.parameters()}, {'params':Encoder.parameters()}], lr=config['attack_lr'])
    elif config['attack_optimizer_type'] == 'SGD':
        Optimizer = optim.SGD([{'params': Attack.parameters()}, {'params':Encoder.parameters()}], lr=config['attack_lr'])
    # Forward and Backward
    for _iter in range(config['attack_max_iter']):
        # zero grad
        Optimizer.zero_grad()
        # attack forward
        attack_seq = Attack(src_seq, refer_seq, attack_index, edge_mask) # 1 x num_frames x height x width
        attack_feat = Encoder(attack_seq)[0] # 1 x num_parts x feat_dim
        # print('Attack: attack_seq={}, attack_feat={}'.format(attack_seq.size(), attack_feat.size()))
        # compute loss
        total_loss_metric = torch.zeros(1).cuda()
        if config['attack_mode'] == 'Targeted' and config['attack_pullclose_weight'] > 0:
            pullclose_loss_metric = Attack_Pullclose_Loss(attack_feat, fake_feat)
            total_loss_metric += pullclose_loss_metric * config['attack_pullclose_weight']
        if config['attack_pushaway_weight'] > 0:
            pushaway_loss_metric = Attack_Pushaway_Loss(attack_feat, src_feat)
            total_loss_metric += pushaway_loss_metric * config['attack_pushaway_weight']
        if config['attack_overlap_weight'] > 0 :
            overlap_loss_metric = Attack_Overlap_Loss(attack_seq[:, attack_index, :, :], refer_seq, edge_mask)
            total_loss_metric += overlap_loss_metric * config['attack_overlap_weight']
        # attack backward
        total_loss_metric.backward()
        # zero grad for non-edge area
        # if config['attack_edge_only']:
        #     Attack.template.grad.masked_fill_(edge_mask < 1e-3, 0)
        # update
        Optimizer.step()
        #############################################################
        # print
        if _iter % 10 == 0 or (_iter+1)==config['attack_max_iter']:
            print('{:#^30}: type={}, lr={:.6f}, weight_decay={:.6f}'.format( \
                'Optimizer', config['attack_optimizer_type'], Optimizer.param_groups[0]['lr'], Optimizer.param_groups[0]['weight_decay']))
            if config['attack_mode'] == 'Targeted' and config['attack_pullclose_weight'] > 0:
                print('{:#^30}: iter={}, pullclose_loss_weight={}, pullclose_loss_metric={}'.format( \
                        'Pullclose Loss', _iter, config['attack_pullclose_weight'], pullclose_loss_metric.data.cpu().numpy()))
            if config['attack_pushaway_weight'] > 0:
                print('{:#^30}: iter={}, pushaway_loss_weight={}, pushaway_loss_metric={}'.format( \
                        'Pushaway Loss', _iter, config['attack_pushaway_weight'], pushaway_loss_metric.data.cpu().numpy()))
            if config['attack_overlap_weight'] > 0:
                print('{:#^30}: iter={}, overlap_loss_weight={}, overlap_loss_metric={}'.format( \
                        'Overlap Loss', _iter, config['attack_overlap_weight'], overlap_loss_metric.data.cpu().numpy()))
            print('{:#^30}: iter={}, total_loss_metric={}'.format( \
                        'Total Loss', _iter, total_loss_metric.data.cpu().numpy()))

    # generate template attack
    des_seq = Attack(src_seq, refer_seq, attack_index, edge_mask) # 1 x num_frames x height x width
    des_feat = Encoder(des_seq)[0] # 1 x num_parts x feat_dim
    des_seq = des_seq.squeeze(0).data.cpu().numpy() # num_frames x height x width
    des_feat =  des_feat.squeeze(0).data.cpu().numpy() # num_parts x feat_dim
    print('Attack: des_seq={}, des_seq_min={}, des_seq_max={}'.format(des_seq.shape, np.min(des_seq), np.max(des_seq)))
    print('Attack: des_seq={}, des_feat={}'.format(des_seq.shape, des_feat.shape))
    return attack_index, des_seq, des_feat

##########################################################################################################################