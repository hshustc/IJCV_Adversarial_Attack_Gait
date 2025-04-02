import torch
import torch.nn as nn
import torch.nn.functional as F

class PartTripletLoss(nn.Module):
    def __init__(self, margin, heter_mining=False, heter_topk=16, hard_mining=False, hard_topk=16, part_mining=False, part_topk=16, nonzero=True, dist_type='euc'):
        super(PartTripletLoss, self).__init__()
        self.margin = margin
        self.heter_mining = heter_mining
        self.heter_topk = heter_topk
        self.hard_mining = hard_mining
        self.hard_topk = hard_topk
        self.part_mining = part_mining
        self.part_topk = part_topk
        self.nonzero = nonzero
        assert(dist_type == 'euc' or dist_type == 'cos')
        self.dist_type = dist_type

    def forward(self, feature, label):
        # feature: [n, m, d], label: [n, m]      
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)

        if self.dist_type == 'euc':
            dist = self.batch_euc_dist(feature)
        elif self.dist_type == 'cos':
            dist = self.batch_cos_dist(feature)
        dist = dist.view(-1)
        
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
        #############################################################
        if self.heter_mining:
            n = 1
            full_hp_dist = torch.topk(full_hp_dist, self.heter_topk, dim=0, largest=True)[0]
            full_hn_dist = torch.topk(full_hn_dist, self.heter_topk, dim=0, largest=False)[0]
            full_hp_dist = torch.mean(full_hp_dist, dim=0, keepdim=True)
            full_hn_dist = torch.mean(full_hn_dist, dim=0, keepdim=True)
        #############################################################
        if self.hard_mining:
            full_hp_dist = torch.topk(full_hp_dist, self.hard_topk, dim=2, largest=True)[0]
            full_hn_dist = torch.topk(full_hn_dist, self.hard_topk, dim=3, largest=False)[0]
        #############################################################
        if self.margin > 0:
            full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)
        else:
            full_loss_metric = F.softplus(full_hp_dist - full_hn_dist).view(n, -1)  

        nonzero_num = (full_loss_metric != 0).sum(1).float()

        if self.nonzero:
            full_loss_metric_mean = full_loss_metric.sum(1) / nonzero_num
            full_loss_metric_mean[nonzero_num == 0] = 0
        else:
            full_loss_metric_mean = full_loss_metric.mean(1)

        #############################################################
        if self.part_mining:
            assert(n > self.part_topk)
            _, hard_part_index = torch.topk(full_loss_metric_mean, k=self.part_topk, dim=0, largest=True)
            full_loss_metric_mean = full_loss_metric_mean[hard_part_index]
            nonzero_num = nonzero_num[hard_part_index]
        #############################################################
        
        # print("full_loss_metric={}, nonzero_num={}".format(full_loss_metric_mean, nonzero_num))
        return full_loss_metric_mean.mean(), nonzero_num.mean()

    def batch_euc_dist(self, x):
        # input: num_parts x batch_size x feat_dim, output: num_parts x batch_size x batch_size
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist
    
    def batch_cos_dist(self, x, scale=8):
        # input: num_parts x batch_size x feat_dim, output: num_parts x batch_size x batch_size
        x2 = F.normalize(x, p=2, dim=2)
        dist = 1 - torch.matmul(x2, x2.transpose(1, 2)) # num_parts x batch_size x batch_size 
        return dist*scale