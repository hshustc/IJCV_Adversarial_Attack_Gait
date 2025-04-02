import os
import os.path as osp
import numpy as np
import pickle
import json
import cv2
from copy import deepcopy
import torch
import torch.nn as nn
import torch.autograd as autograd

# Miscellaneous
def save_sil(seq, des_path, attack_index=None):
    if not osp.exists(des_path):
        os.makedirs(des_path)
    num_frames = seq.shape[0]
    for i in range(num_frames):
        sil = seq[i, :, :]
        if (attack_index is not None) and (i in attack_index):
            des_name = '{:0>3d}_attack.png'.format(i)
        else:
            des_name = '{:0>3d}.png'.format(i)
        cv2.imwrite(osp.join(des_path, des_name), sil)

def save_pkl(seq, des_path, attack_index=None):
    if not osp.exists(des_path):
        os.makedirs(des_path)
    pkl_name = osp.join(des_path, '{}.pkl'.format(osp.basename(des_path)))
    with open(pkl_name, 'wb') as f:
        pickle.dump(seq, f)

# Dataset
def split_dataset(source, dataname):
    assert(dataname in ['CASIA_B', 'OUMVLP', 'Gait3D', 'GREW'])
    probe_seq_dict = {'CASIA_B': ['nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02'],
                      'OUMVLP': ['00'],
                      }
    gallery_seq_dict = {'CASIA_B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                        'OUMVLP': ['01'],
                        }
    probe_path_list = list()
    gallery_path_list = list()
    for idx, path in enumerate(source.seq_dir_list):
        _id, _type, _view = path.split('/')[-3:]
        assert(_id == source.seq_label_list[idx])
        if dataname in ['CASIA_B', 'OUMVLP']:
            if _type in probe_seq_dict[dataname]:
                probe_path_list.append(path)
            elif _type in gallery_seq_dict[dataname]:
                gallery_path_list.append(path)
            else:
                print('Unknown Type {} for Dataset {}'.format(_type, dataname))
                os._exit(1)
        elif dataname in ['Gait3D', 'GREW']:
            pid_json_dict = {'Gait3D': 'partition/Gait3D.json',
                            'GREW': 'partition/GREW.json',
                        }
            pid_json = pid_json_dict[dataname]
            with open(pid_json, 'rb') as f:
                pid_list = json.load(f)
            probe_set = pid_list['PROBE_SET']
            seq_label_type_view = '-'.join([_id, _type, _view])
            if seq_label_type_view in probe_set:
                probe_path_list.append(path)
            else:
                gallery_path_list.append(path)
        else:
            print('Unknown Dataset {}'.format(dataname))
            os._exit(1)
    return probe_path_list, gallery_path_list

# Attack

def np2var(x, requires_grad=False, use_gpu=True):
    x = torch.from_numpy(x)
    if use_gpu:
        x = x.cuda()
    x = autograd.Variable(x, requires_grad=requires_grad)
    return x

def unidirectional_edge_mask(binary_sil, edge_thres):
    dist = cv2.distanceTransform(binary_sil, cv2.DIST_L2, 3)  # the closest distance to background
    mask = (dist > 0) & (dist <= edge_thres)
    return mask

def bidirectional_edge_mask(sil, edge_thres):
    if np.max(sil) > 1.0:
        sil = sil.astype('uint8')
    else:
        sil = (sil*255.0).astype('uint8')
    # print('bidirectional_edge_mask: sil_size={}, sil_min={}, sil_max={}'.format(sil.shape, np.min(sil), np.max(sil)))
    assert(np.max(sil) == 255)
    _, binary_sil = cv2.threshold(sil, 128, 255, cv2.THRESH_BINARY)
    mask1 = unidirectional_edge_mask(binary_sil, edge_thres)
    mask2 = unidirectional_edge_mask(255-binary_sil, edge_thres)
    mask = mask1 | mask2
    return mask.astype('int')

def make_path_id(path):
    _id = path.split('/')[-3]
    return _id

def make_path_type(path):
    _type = path.split('/')[-2]
    return _type

def make_path_view(path):
    _id = path.split('/')[-1]
    return _id

def make_path_key(path):
    _id, _type, _view = path.split('/')[-3:]
    return '-'.join([_id, _type, _view])

def make_basic_attack_info(source, probe_path_list, gallery_path_list, targeted_attack=False, cross_view_eval=False): # num_fake_gallery=1
    attack_info = dict()
    for idx, path in enumerate(probe_path_list):
        probe_key = make_path_key(path)
        sub_attack_info = dict()
        sub_attack_info.update({'seq_path':path})
        if not targeted_attack:
            sub_attack_info.update({'fake_view':None})
            sub_attack_info.update({'fake_id':None})
            sub_attack_info.update({'fake_seq_path':None})
        else:
            all_fake_view = list()
            all_fake_id = list()
            all_fake_seq_path = list()
            if not cross_view_eval:
                # fake view
                fake_view = 'all'
                all_fake_view.append(fake_view)
                # fake_id
                probe_id = make_path_id(path)
                fake_id_list = [make_path_id(_) for _ in gallery_path_list]
                fake_id_list = sorted(list(set(fake_id_list)))
                if probe_id in fake_id_list:
                    fake_id_list.remove(probe_id)
                fake_id_index = np.random.choice(np.arange(len(fake_id_list)), 1, replace=False)[0]
                fake_id = fake_id_list[fake_id_index]
                assert(fake_id != probe_id)
                all_fake_id.append(fake_id)
                # fake_seq_path
                fake_seq_path_list = [_ for _ in gallery_path_list if make_path_id(_) == fake_id]
                assert(len(fake_seq_path_list) > 0)
                # if len(fake_seq_path_list) > num_fake_gallery:
                #     fake_seq_path_index = np.random.choice(np.arange(len(fake_seq_path_list)), num_fake_gallery, replace=False)
                # else:
                #     fake_seq_path_index = np.arange(len(fake_seq_path_list))
                fake_seq_path_index = np.arange(len(fake_seq_path_list))
                np.random.shuffle(fake_seq_path_index)
                fake_seq_path = [fake_seq_path_list[i] for i in fake_seq_path_index]
                all_fake_seq_path.append(fake_seq_path)
            else:
                view_list = [make_path_view(_) for _ in gallery_path_list]
                view_list = sorted(list(set(view_list)))
                for view in view_list:
                    # fake_view
                    fake_view = view
                    all_fake_view.append(fake_view)
                    # fake id
                    probe_id = make_path_id(path)
                    fake_id_list = [make_path_id(_) for _ in gallery_path_list if make_path_view(_)==view]
                    fake_id_list = sorted(list(set(fake_id_list)))
                    if probe_id in fake_id_list:
                        fake_id_list.remove(probe_id)
                    fake_id_index = np.random.choice(np.arange(len(fake_id_list)), 1, replace=False)[0]
                    fake_id = fake_id_list[fake_id_index]
                    assert(fake_id != probe_id)
                    all_fake_id.append(fake_id)
                    # fake_seq_path
                    fake_seq_path_list = [_ for _ in gallery_path_list if make_path_id(_)==fake_id and make_path_view(_) == fake_view]
                    assert(len(fake_seq_path_list) > 0)
                    # if len(fake_seq_path_list) > num_fake_gallery:
                    #     fake_seq_path_index = np.random.choice(np.arange(len(fake_seq_path_list)), num_fake_gallery, replace=False)
                    # else:
                    #     fake_seq_path_index = np.arange(len(fake_seq_path_list))
                    fake_seq_path_index = np.arange(len(fake_seq_path_list))
                    np.random.shuffle(fake_seq_path_index)
                    fake_seq_path = [fake_seq_path_list[i] for i in fake_seq_path_index]
                    all_fake_seq_path.append(fake_seq_path)
            sub_attack_info.update({'fake_view':all_fake_view})
            sub_attack_info.update({'fake_id':all_fake_id})
            sub_attack_info.update({'fake_seq_path':all_fake_seq_path})
        attack_info.update({probe_key:sub_attack_info})
        print('probe_key={}, fake_view={}, fake_id={}, fake_seq_path={}'.format(probe_key, \
                sub_attack_info['fake_view'], sub_attack_info['fake_id'], sub_attack_info['fake_seq_path']))
    return attack_info
            