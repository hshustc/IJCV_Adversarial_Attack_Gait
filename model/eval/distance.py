import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np

'''
def cuda_euc_dist(x, y):
    x = x.permute(1, 0, 2).contiguous() # num_parts * num_probe * part_dim
    y = y.permute(1, 0, 2).contiguous() # num_parts * num_gallery * part_dim
    dist = torch.sum(x ** 2, 2).unsqueeze(2) + torch.sum(y ** 2, 2).unsqueeze(
        2).transpose(1, 2) - 2 * torch.matmul(x, y.transpose(1, 2)) # num_parts * num_probe * num_gallery
    dist = torch.sqrt(F.relu(dist)) # num_parts * num_probe * num_gallery
    dist = torch.mean(dist, 0) # num_probe * num_gallery
    return dist

def cuda_cos_dist(x, y):
    x = F.normalize(x, p=2, dim=2).permute(1, 0, 2) # num_parts * num_probe * part_dim
    y = F.normalize(y, p=2, dim=2).permute(1, 0, 2) # num_parts * num_gallery * part_dim
    dist = 1 - torch.mean(torch.matmul(x, y.transpose(1, 2)), 0) # num_probe * num_gallery
    return dist
'''

def cuda_dist(x, y, config, part_average=True, chunk=5000):
    # x/y: num_probe/num_gallery * num_parts * part_dim
    num_probe, num_parts = x.size(0), x.size(1)
    num_gallery = y.size(0)
    assert(num_parts == y.size(1))
    dist = torch.zeros(num_probe, num_gallery)
    for p_start in range(0, num_probe, chunk):
        for g_start in range(0, num_gallery, chunk):
            p_end = p_start+chunk if p_start+chunk < num_probe else num_probe
            g_end = g_start+chunk if g_start+chunk < num_gallery else num_gallery
            chunk_x = x[p_start:p_end, :, :] # chunk * num_parts * part_dim
            chunk_y = y[g_start:g_end, :, :] # chunk * num_parts * part_dim
            if not part_average:
                chunk_x = chunk_x.view(chunk_x.size(0), -1) # chunk * (num_parts * part_dim)
                chunk_y = chunk_y.view(chunk_y.size(0), -1) # chunk * (num_parts * part_dim)
            if config['dist_type'] == 'euc':
                chunk_dist = cuda_cal_euc_dist(chunk_x, chunk_y)
            elif config['dist_type'] == 'cos':
                chunk_dist = cuda_cal_cos_dist(chunk_x, chunk_y)
            elif config['dist_type'] == 'emd':
                chunk_dist = cuda_cal_emd_dist(chunk_x, chunk_y, \
                    config['emd_base'], config['emd_topk'], config['emd_method'], config['emd_reverse'], config['emd_lambda'], \
                    config['emd_part_window'], config['emd_flow_window'])
            dist[p_start:p_end, g_start:g_end] = chunk_dist    
            del chunk_x, chunk_y, chunk_dist
    return dist

def cuda_cal_euc_dist(x, y, average=True):
    assert(x.dim() == y.dim())
    assert(x.dim() == 2 or x.dim() == 3)
    if x.dim() == 3:
        # x: num_probe x num_parts x feat_dim, y: num_gallery x num_parts x feat_dim
        x = x.permute(1, 0, 2).contiguous() # num_parts x num_probe x feat_dim
        y = y.permute(1, 0, 2).contiguous() # num_parts x num_gallery x feat_dim
        dist = torch.sum(x ** 2, 2).unsqueeze(2) + torch.sum(y ** 2, 2).unsqueeze(
            2).transpose(1, 2) - 2 * torch.matmul(x, y.transpose(1, 2)) # num_parts x num_probe x num_gallery
        dist = torch.sqrt(F.relu(dist)) # num_parts x num_probe x num_gallery
        if average:
            dist = torch.mean(dist, 0) # num_probe x num_gallery
    else:
        # x: num_probe x feat_dim, y: num_gallery x feat_dim
        dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
            1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1)) # num_probe x num_gallery
        dist = torch.sqrt(F.relu(dist)) # num_probe x num_gallery
    return dist

def cuda_cal_cos_dist(x, y, average=True):
    assert(x.dim() == y.dim())
    assert(x.dim() == 2 or x.dim() == 3)
    if x.dim() == 3:
        # x: num_probe x num_parts x feat_dim, y: num_gallery x num_parts x feat_dim
        x = x.permute(1, 0, 2).contiguous() # num_parts x num_probe x feat_dim
        y = y.permute(1, 0, 2).contiguous() # num_parts x num_gallery x feat_dim        
        x = F.normalize(x, p=2, dim=2) # num_parts x num_probe x feat_dim
        y = F.normalize(y, p=2, dim=2) # num_parts x num_gallery x feat_dim
        dist = 1 - torch.matmul(x, y.transpose(1, 2)) # num_parts x num_probe x num_gallery
        if average:
            dist = torch.mean(dist, 0) # num_probe x num_gallery    
    else:
        # x: num_probe x feat_dim, y: num_gallery x feat_dim
        x = F.normalize(x, p=2, dim=1) # num_probe x feat_dim
        y = F.normalize(y, p=2, dim=1) # num_gallery x feat_dim
        dist = 1 - torch.matmul(x, y.transpose(0, 1)) # num_probe x num_gallery
    return dist

def cuda_cal_emd_dist(x, y, emd_base='cos', emd_topk=10, emd_method='uniform', emd_reverse=False, emd_lambda=0.05, \
                        emd_part_window=-1, emd_flow_window=3):
    # x: num_probe x num_parts x feat_dim, y: num_gallery x num_parts x feat_dim
    assert(x.dim() == y.dim())
    assert(x.dim() == 3)
    assert(emd_base == 'cos' or emd_base == 'euc')
    #############################################################
    # dist
    if emd_base == 'cos':
        xy_part_dist = cuda_cal_cos_dist(x, y, average=False) # num_parts x num_probe x num_gallery
    elif emd_base == 'euc':
        xy_part_dist = cuda_cal_euc_dist(x, y, average=False) # num_parts x num_probe x num_gallery
    xy_dist = torch.mean(xy_part_dist, 0) # num_probe x num_gallery
    # jaccard_topk
    jaccard_topk = int(emd_topk * 1.0)
    _, xy_topk = torch.topk(xy_dist, jaccard_topk, dim=-1, largest=False) # num_probe x jaccard_topk
    if emd_method == 'jaccard':
        _, xy_part_topk = torch.topk(xy_part_dist, max(emd_topk, jaccard_topk), dim=-1, largest=False) # num_parts x num_probe x max(emd_topk, jaccard_topk)
        # dist
        if emd_base == 'cos':
            yy_part_dist = cuda_cal_cos_dist(y, y, average=False) # num_parts x num_gallery x num_gallery
        elif emd_base == 'euc':
            yy_part_dist = cuda_cal_euc_dist(y, y, average=False) # num_parts x num_gallery x num_gallery
        yy_dist = torch.mean(yy_part_dist, 0) # num_gallery x num_gallery
        # jaccard_topk
        _, yy_topk = torch.topk(yy_dist, jaccard_topk, dim=-1, largest=False) # num_gallery x jaccard_topk
        _, yy_part_topk = torch.topk(yy_part_dist, jaccard_topk, dim=-1, largest=False) # num_parts x num_gallery x jaccard_topk
    #############################################################
    num_probe = x.size(0)
    xy_dist = torch.ones_like(xy_dist) * 100 # reset xy_dist
    for i in range(num_probe):
        anchor = x[i, :, :] # num_parts x feat_dim
        anchor_topk = xy_topk[i, :jaccard_topk] # jaccard_topk
        emd_index = xy_topk[i, :emd_topk] # emd_topk
        gallery = y[emd_index, :, :] # emd_topk x num_parts x feat_dim
        if emd_method == 'jaccard':
            anchor_part_topk = xy_part_topk[:, i, :] # num_parts x jaccard_topk
            gallery_topk = yy_topk[emd_index, :] # emd_topk x jaccard_topk
            gallery_part_topk = yy_part_topk[:, emd_index, :] # num_parts x emd_topk x jaccard_topk
            anchor_emd_dist = one2all_emd_dist(anchor, gallery, emd_base, emd_method, emd_reverse, emd_lambda, emd_part_window, emd_flow_window, \
                                                anchor_topk, anchor_part_topk, gallery_topk, gallery_part_topk) # topk
        else:
            anchor_emd_dist = one2all_emd_dist(anchor, gallery, emd_base, emd_method, emd_reverse, emd_lambda, emd_part_window, emd_flow_window) # topk
        xy_dist[i][emd_index] = anchor_emd_dist # topk
    return xy_dist

def one2all_emd_dist(anchor, gallery, emd_base, emd_method, emd_reverse, emd_lambda, emd_part_window, emd_flow_window, \
                        anchor_topk=None, anchor_part_topk=None, gallery_topk=None, gallery_part_topk=None):
    # anchor: num_parts x feat_dim, gallery: num_gallery (emd_topk) x num_parts x feat_dim
    # anchor_topk: jaccard_topk, anchor_part_topk: num_parts x jaccard_topk
    # gallery_topk: num_gallery (emd_topk) x jaccard_topk, gallery_part_topk: num_parts x num_gallery (emd_topk) x jaccard_topk
    N, R, _ = gallery.size()
    #############################################################
    re_anchor = anchor.unsqueeze(0).repeat(N, 1, 1) # num_gallery x num_parts x feat_dim
    if emd_base == 'cos':
        dist = cuda_cal_cos_dist(gallery.permute(1, 0, 2).contiguous(), re_anchor.permute(1, 0, 2).contiguous(), average=False) # num_gallery x num_parts x num_parts
        norm_dist = dist # num_gallery x num_parts x num_parts
        sim = 1 - norm_dist # num_gallery x num_parts x num_parts
    elif emd_base == 'euc':
        dist = cuda_cal_euc_dist(gallery.permute(1, 0, 2).contiguous(), re_anchor.permute(1, 0, 2).contiguous(), average=False) # num_gallery x num_parts x num_parts
        norm_dist = Eucnorm(dist) # num_gallery x num_parts x num_parts
        sim = 1 - norm_dist # num_gallery x num_parts x num_parts
    #############################################################
    if emd_method == 'uniform':
        u = torch.zeros(N, R, dtype=dist.dtype, device=dist.device).fill_(1. / R) # num_gallery x num_parts
        v = torch.zeros(N, R, dtype=dist.dtype, device=dist.device).fill_(1. / R) # num_gallery x num_parts
        print('u={}, norm_u={}'.format(u[0], u[0]/u[0].sum()))
        print('v={}, norm_v={}'.format(v[0], v[0]/v[0].sum()))
        u = u / (u.sum(dim=1, keepdims=True) + 1e-7) # num_gallery x num_parts  
        v = v / (v.sum(dim=1, keepdims=True) + 1e-7) # num_gallery x num_parts
    elif emd_method == 'meansim' or emd_method == 'maxsim':
        #############################################################
        if emd_part_window > 0:
            part_mask = Adjmask(num_parts=R, window_size=emd_part_window).to(anchor.device) # num_parts x num_parts
        else:
            part_mask = torch.ones((R, R)).to(anchor.device) # num_parts x num_parts
        part_mask = part_mask.unsqueeze(0).repeat(N, 1, 1) # num_gallery x num_parts x num_parts
        #############################################################
        if emd_reverse:
            pos_sim = F.relu(1 - sim)
        else:
            pos_sim = F.relu(sim)
        if emd_method == 'meansim':
            u = torch.mean(pos_sim * part_mask, 2) # num_gallery x num_parts 
            v = torch.mean(pos_sim.permute(0, 2, 1) * part_mask, 2) # num_gallery x num_parts
        elif emd_method == 'maxsim':
            u = torch.max(pos_sim * part_mask, 2)[0] # num_gallery x num_parts
            v = torch.max(pos_sim.permute(0, 2, 1) * part_mask, 2)[0] # num_gallery x num_parts 
        print('u={}, norm_u={}'.format(u[0], u[0]/u[0].sum()))
        print('v={}, norm_v={}'.format(v[0], v[0]/v[0].sum()))
        u = u / (u.sum(dim=1, keepdims=True) + 1e-7) # num_gallery x num_parts  
        v = v / (v.sum(dim=1, keepdims=True) + 1e-7) # num_gallery x num_parts
    elif emd_method == 'meanfeat' or emd_method == 'maxfeat':
        if emd_method == 'meanfeat':
            global_anchor = torch.mean(re_anchor, dim=1, keepdims=True) # num_gallery x 1 x feat_dim
            global_gallery = torch.mean(gallery, dim=1, keepdims=True) # num_gallery x 1 x feat_dim
        elif emd_method == 'maxfeat':
            global_anchor = torch.max(re_anchor, dim=1, keepdims=True)[0] # num_gallery x 1 x feat_dim
            global_gallery = torch.max(gallery, dim=1, keepdims=True)[0] # num_gallery x 1 x feat_dim
        if emd_base == 'cos':
            u = cuda_cal_cos_dist(gallery.permute(1, 0, 2).contiguous(), global_anchor.permute(1, 0, 2).contiguous(), average=False) # num_gallery x num_parts x 1
            v = cuda_cal_cos_dist(re_anchor.permute(1, 0, 2).contiguous(), global_gallery.permute(1, 0, 2).contiguous(), average=False) # num_gallery x num_parts x 1
            if emd_reverse:
                u = F.relu(u.squeeze()) # num_gallery x num_parts
                v = F.relu(v.squeeze()) # num_gallery x num_parts
            else:
                u = F.relu(1 - u.squeeze()) # num_gallery x num_parts
                v = F.relu(1 - v.squeeze()) # num_gallery x num_parts
        elif emd_base == 'euc':
            u = cuda_cal_euc_dist(gallery.permute(1, 0, 2).contiguous(), global_anchor.permute(1, 0, 2).contiguous(), average=False) # num_gallery x num_parts x 1
            v = cuda_cal_euc_dist(re_anchor.permute(1, 0, 2).contiguous(), global_gallery.permute(1, 0, 2).contiguous(), average=False) # num_gallery x num_parts x 1
            if emd_reverse:
                u = F.relu(Eucnorm(u.squeeze())) # num_gallery x num_parts
                v = F.relu(Eucnorm(v.squeeze())) # num_gallery x num_parts
            else:                
                u = F.relu(1 - Eucnorm(u.squeeze())) # num_gallery x num_parts
                v = F.relu(1 - Eucnorm(v.squeeze())) # num_gallery x num_parts
        print('u={}, norm_u={}'.format(u[0], u[0]/u[0].sum()))
        print('v={}, norm_v={}'.format(v[0], v[0]/v[0].sum()))
        u = u / (u.sum(dim=1, keepdims=True) + 1e-7) # num_gallery x num_parts 
        v = v / (v.sum(dim=1, keepdims=True) + 1e-7) # num_gallery x num_parts
    elif emd_method == 'jaccard':
        u = torch.zeros(N, R, dtype=dist.dtype, device=dist.device) # num_gallery x num_parts
        v = torch.zeros(N, R, dtype=dist.dtype, device=dist.device) # num_gallery x num_parts
        num_parts, num_gallery, jaccard_topk = gallery_part_topk.size()
        for i in range(num_gallery):
            for j in range(num_parts):
                # u: gallery_part_topk v.s. anchor_topk, we have at least one duplicate
                u_unique = torch.unique(torch.cat((gallery_part_topk[j, i], anchor_topk))) 
                u[i][j] = (2*jaccard_topk - len(u_unique) + 1) / (len(u_unique) + 1)
                # v: gallery_topk v.s. anchor_part_topk, we do not always have duplicate
                v_unique = torch.unique(torch.cat((gallery_topk[i, :], anchor_part_topk[j, :])))
                v[i][j] = (2*jaccard_topk - len(v_unique) + 1) / (len(v_unique) + 1)
            # print('u={}, norm_u={}'.format(u[i], u[i]/u[i].sum()))
            # print('v={}, norm_v={}'.format(v[i], v[i]/v[i].sum()))
        print('u={}, norm_u={}'.format(u[0], u[0]/u[0].sum()))
        print('v={}, norm_v={}'.format(v[0], v[0]/v[0].sum()))
        u = u / (u.sum(dim=1, keepdims=True) + 1e-7) # num_gallery x num_parts 
        v = v / (v.sum(dim=1, keepdims=True) + 1e-7) # num_gallery x num_parts
    else:
        print('Illegal EMD Method')
        exit(0)
    #############################################################
    if emd_flow_window > 0:
        flow_mask = Adjmask(num_parts=R, window_size=emd_flow_window).to(anchor.device) # num_parts x num_parts
    else:
        flow_mask = torch.ones((R, R)).to(anchor.device) # num_parts x num_parts
    flow_mask = flow_mask.unsqueeze(0).repeat(N, 1, 1) # num_gallery x num_parts x num_parts
    norm_dist[flow_mask==0] = 100 # num_gallery x num_parts x num_parts
    K = torch.exp(-norm_dist / emd_lambda)  # num_gallery x num_parts x num_parts
    T = Sinkhorn(K, u, v) # num_gallery x num_parts x num_parts
    print('T={}'.format(T[0]))
    if torch.any(torch.isnan(T)) or torch.any(torch.isinf(T)):
        print('Illegal T with nan or inf')
        exit(0)
    dist = torch.sum(T * dist, dim=(1, 2)) # num_gallery
    return dist

def Sinkhorn(K, u, v):
    # K: num_gallery x num_parts x num_parts, u: num_gallery x num_parts, v: num_gallery x num_parts
    r = torch.ones_like(u)  # num_gallery x num_parts
    c = torch.ones_like(v)  # num_gallery x num_parts
    thresh = 1e-1
    for it in range(100):
        r0 = r
        r = u / (torch.matmul(K, c.unsqueeze(-1)).squeeze(-1) + 1e-7) # numerator/denominator: num_gallery x num_parts
        c = v / (torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1) + 1e-7) # numerator/denominator: num_gallery x num_parts
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K # num_gallery x num_parts x num_parts
    return T

def Adjmask(num_parts, window_size):
    e = torch.ones((num_parts, num_parts))
    k = int((window_size - 1) / 2)
    mask = torch.tril(e, k) * torch.triu(e, -k)
    return mask

# from AlignReID
def Eucnorm(x, scale=1):
    return (torch.exp(x/scale) - 1) / (torch.exp(x/scale) + 1)