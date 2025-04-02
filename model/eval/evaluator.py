import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import json
import pickle
from .re_ranking import re_ranking
from .metric import compute_CMC_mAP, compute_PR
from .distance import cuda_dist

def make_attack_data(data, config, gallery_view=None):
    real_feature, real_view, real_seq_type, real_label = data
    if config['attack_info'] is None:
        return real_feature, np.asarray(real_label)
    with open(config['attack_info'], 'rb') as f:
        attack_info  = pickle.load(f)
    if config['attack_mode'] =='Untargeted':
        fake_feature = list()
        cnt_real, cnt_fake = 0, 0
        for i in range(len(real_label)):
            seq_key = '-'.join([real_label[i], real_seq_type[i], real_view[i]])
            if seq_key in attack_info.keys():
                sub_attack_info = attack_info[seq_key]
                fake_feature.append(torch.from_numpy(sub_attack_info['attack_feat'][0]).to(real_feature.device))
                cnt_fake += 1
            else:
                fake_feature.append(real_feature[i])
                cnt_real += 1
        fake_feature = torch.stack(fake_feature, dim=0)
        print('cnt_total={}, cnt_real={}, cnt_fake={}, gallery_view={}'.format(len(real_label), cnt_real, cnt_fake, gallery_view))
        return fake_feature, np.asarray(real_label)
    elif config['attack_mode'] =='Targeted':
        fake_feature, fake_label = list(), list()
        cnt_real, cnt_fake = 0, 0
        dataname = config['dataset'][0]
        if dataname in ['CASIA_B', 'OUMVLP']:
            for i in range(len(real_label)):
                seq_key = '-'.join([real_label[i], real_seq_type[i], real_view[i]])
                if seq_key in attack_info.keys():
                    sub_attack_info = attack_info[seq_key]
                    view_index = sub_attack_info['fake_view'].index(gallery_view)
                    fake_feature.append(torch.from_numpy(sub_attack_info['attack_feat'][view_index]).to(real_feature.device))
                    fake_label.append(sub_attack_info['fake_id'][view_index])
                    cnt_fake += 1
                else:
                    fake_feature.append(real_feature[i])
                    fake_label.append(real_label[i])
                    cnt_real += 1
        elif dataname in ['Gait3D', 'GREW']:
            for i in range(len(real_label)):
                seq_key = '-'.join([real_label[i], real_seq_type[i], real_view[i]])
                if seq_key in attack_info.keys():
                    sub_attack_info = attack_info[seq_key]
                    fake_feature.append(torch.from_numpy(sub_attack_info['attack_feat'][0]).to(real_feature.device))
                    fake_label.append(sub_attack_info['fake_id'][0])
                    cnt_fake += 1
                else:
                    fake_feature.append(real_feature[i])
                    fake_label.append(real_label[i])
                    cnt_real += 1
        fake_feature = torch.stack(fake_feature, dim=0)
        print('cnt_total={}, cnt_real={}, cnt_fake={}, gallery_view={}'.format(len(real_label), cnt_real, cnt_fake, gallery_view))
        return fake_feature, np.asarray(fake_label)

def tensor_select_feature(feature, mask):
    return feature[mask, :, :]

def list_select_feature(feature, mask):
    output = [feature[i] for i in range(len(mask)) if mask[i] > 0]
    return output

def evaluation(data, config):
    print("####################################################")
    print('Compute {} Distance'.format(config['dist_type'].upper()))
    if config['dist_type'].upper() == 'EMD':
        print('EMD_TopK={}, EMD_Lambda={}, EMD_Method={}'.format(config['emd_topk'], config['emd_lambda'], config['emd_method']))
    print("####################################################")

    # indoor
    indoor_probe_seq_dict = {'CASIA_B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']],
                      }
    indoor_gallery_seq_dict = {'CASIA_B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        }

    #############################################################
    config.update({'select_func':tensor_select_feature})
    config.update({'dist_func':cuda_dist})
    #############################################################
    
    dataset = config['dataset'][0].replace('-', '_')
    if dataset in indoor_gallery_seq_dict.keys():
        return evaluation_indoor(data, config, indoor_probe_seq_dict, indoor_gallery_seq_dict)
    else:
        print('Illegal Dataset Type')
        os._exit(0)
   
def evaluation_indoor(data, config, probe_seq_dict, gallery_seq_dict):
    dataset = config['dataset'][0].replace('-', '_')
    feature, view, seq_type, label = data
    label = np.asarray(label)
    view_list = sorted(list(set(view)))
    view_num = len(view_list)
    sample_num = len(feature)
    print("Feature Shape: ", feature.shape)

    #############################################################
    if config['attack_mode'] == 'Untargeted':
           feature, label = make_attack_data(data, config)
    #############################################################

    CMC = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, config['max_rank']])
    mAP = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num])
    P_thres = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num])
    R_thres = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    #############################################################
                    if config['attack_mode'] == 'Targeted':
                        feature, label = make_attack_data(data, config, gallery_view=gallery_view)
                    #############################################################
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_y = label[gseq_mask]
                    gseq_mask = torch.from_numpy(np.asarray(gseq_mask, dtype=np.uint8))
                    # gallery_x = feature[gseq_mask, :, :]
                    gallery_x = config['select_func'](feature, gseq_mask)

                    if config['remove_no_gallery']:
                        pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view]) & np.isin(label, gallery_y)
                    else:
                        pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_y = label[pseq_mask]
                    pseq_mask = torch.from_numpy(np.asarray(pseq_mask, dtype=np.uint8))
                    # probe_x = feature[pseq_mask, :, :]
                    probe_x = config['select_func'](feature, pseq_mask)

                    if config['reranking']:
                        assert(config['dist_type'] == 'cos')
                        dist_p_p = 1 - config['dist_func'](probe_x, probe_x, config).cpu().numpy()
                        dist_p_g = 1 - config['dist_func'](probe_x, gallery_x, config).cpu().numpy()
                        dist_g_g = 1 - config['dist_func'](gallery_x, gallery_x, config).cpu().numpy()
                        dist = re_ranking(dist_p_g, dist_p_p, dist_g_g, lambda_value=config['relambda'])
                    else:
                        dist = config['dist_func'](probe_x, gallery_x, config).cpu().numpy()
                    print('Distance Shape: ', dist.shape)
                    eval_results = compute_CMC_mAP(dist, probe_y, gallery_y, config['max_rank'])
                    CMC[p, v1, v2, :] = np.round(eval_results[0] * 100, 2)
                    mAP[p, v1, v2] = np.round(eval_results[1] * 100, 2)
                    if config['dist_type'] == 'cos' and config['cos_sim_thres'] > -1:
                        eval_results = compute_PR(dist, probe_y, gallery_y, config['cos_sim_thres'])
                        P_thres[p, v1, v2] = np.round(eval_results[0] * 100, 2)
                        R_thres[p, v1, v2] = np.round(eval_results[1] * 100, 2)

    return CMC, mAP, [P_thres, R_thres]
