import os
import os.path as osp

import numpy as np
import pickle
import json

from .data_set import DataSet

def load_data(config):
    print("####################################################")
    dataset, dataset_path, resolution, pid_num, pid_shuffle = \
        config['dataset'], config['dataset_path'], config['resolution'], config['pid_num'], config['pid_shuffle']
    print("dataset={}, dataset_path={}, resolution={}".format(dataset, dataset_path, resolution))
    print("####################################################")
    seq_dir_list = list()
    seq_id_list = list()

    cut_padding = None
    for i, dataset_path_i in enumerate(dataset_path):
        dataset_i = dataset[i].replace('-', '_')
        dataset_prefix = "ds{}_".format(i+1) if i > 0 else ''
        print("dataset={}, dataset_path={}, dataset_prefix={}".format(dataset_i, dataset_path_i, dataset_prefix))
        assert(dataset_i.lower().replace('-', '_') in dataset_path_i.lower().replace('-', '_'))
        check_resolution = True
        for _id in sorted(list(os.listdir(dataset_path_i))):
            # In CASIA-B, data of subject #5 is incomplete. Thus, we ignore it in training.
            if dataset_i == 'CASIA_B' and _id == '005':
                continue
            id_path = osp.join(dataset_path_i, _id)
            for _type in sorted(list(os.listdir(id_path))):
                type_path = osp.join(id_path, _type)
                for _view in sorted(list(os.listdir(type_path))):
                    view_path = osp.join(type_path, _view)
                    #############################################################
                    if config['check_frames']:
                        pkl_name = '{}.pkl'.format(os.path.basename(view_path))
                        all_imgs = pickle.load(open(osp.join(view_path, pkl_name), 'rb'))
                        if all_imgs.shape[0] < 15:
                            continue
                    #############################################################
                    if check_resolution:
                        pkl_name = '{}.pkl'.format(os.path.basename(view_path))
                        all_imgs = pickle.load(open(osp.join(view_path, pkl_name), 'rb'))
                        if cut_padding is None: # assign cut_padding for the first time
                            if all_imgs.shape[2]==resolution:
                                cut_padding = int(float(resolution)/64*10)
                            else:
                                cut_padding = 0
                        assert(all_imgs.shape[1]==resolution)
                        if cut_padding > 0: # check width for all datasets
                            assert(all_imgs.shape[2]==resolution)
                        else:
                            assert(all_imgs.shape[2]==(resolution-2*int(float(resolution)/64*10)))
                        check_resolution = False
                        print("Check Resolution: view_path={}, resolution={}, cut_padding={}, img_shape={}".format(\
                                view_path, resolution, cut_padding, all_imgs.shape))
                    #############################################################
                    seq_dir_list.append(view_path)
                    seq_id_list.append(dataset_prefix+_id)

    total_id = len(list(set(seq_id_list)))
    if config['pid_json'] is not None:
        pid_fname = config['pid_json']
        with open(pid_fname, 'rb') as f:
            pid_list = json.load(f)
        train_id_list = pid_list['TRAIN_SET']
        test_id_list = pid_list['TEST_SET']
    else:
        pid_fname = osp.join('partition', '{}_{}_{}_Total_{}.npy'.format(
            '_'.join(dataset), pid_num, pid_shuffle, total_id))
        if not osp.exists(pid_fname):
            pid_list = sorted(list(set(seq_id_list)))
            if pid_shuffle:
                np.random.shuffle(pid_list)
            assert(pid_num <= len(pid_list))
            if pid_num == 0:
                #the first three for training (only for convenience) and all for test
                pid_list = [pid_list[0:3], pid_list[:]]
            elif pid_num == -1: 
                #all for training and the last three for test (only for convenience)
                pid_list = [pid_list[:], pid_list[-3:]]
            else:
                pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
            os.makedirs('partition', exist_ok=True)
            np.save(pid_fname, pid_list)

        pid_list = np.load(pid_fname)
        train_id_list = pid_list[0]
        test_id_list = pid_list[1]

    # train source
    train_seq_dir_list = [seq_dir_list[i] for i, l in enumerate(seq_id_list) if l in train_id_list]
    train_seq_id_list = [seq_id_list[i] for i, l in enumerate(seq_id_list) if l in train_id_list]
    train_index_info = {}
    for i, _id in enumerate(train_seq_id_list):
        if _id not in train_index_info.keys():
            train_index_info[_id] = {}
        _dir = train_seq_dir_list[i]
        _type = _dir.split('/')[-2]
        if _type not in train_index_info[_id].keys():
            train_index_info[_id][_type] = []
        train_index_info[_id][_type].append(i)
    train_source = DataSet(train_seq_dir_list, train_seq_id_list, train_index_info, resolution, cut_padding)
    # test source
    test_seq_dir_list = [seq_dir_list[i] for i, l in enumerate(seq_id_list) if l in test_id_list]
    test_seq_id_list = [seq_id_list[i] for i, l in enumerate(seq_id_list) if l in test_id_list]
    test_index_info = {}
    for i, _id in enumerate(test_seq_id_list):
        if _id not in test_index_info.keys():
            test_index_info[_id] = {}
        _dir = test_seq_dir_list[i]
        _type = _dir.split('/')[-2]
        if _type not in test_index_info[_id].keys():
            test_index_info[_id][_type] = []
        test_index_info[_id][_type].append(i)
    test_source = DataSet(test_seq_dir_list, test_seq_id_list, test_index_info, resolution, cut_padding)

    print("####################################################")
    print("pid_fname:", pid_fname)
    print("resolution={}, cut_padding={}".format(resolution, cut_padding))

    print("number of ids for train:", len(train_id_list))
    print_num = min(30, len(train_id_list))
    print("example ids for train:", train_id_list[0:print_num])
    print("number of seqs for train:", len(train_seq_dir_list))

    print("number of ids for test:", len(test_id_list))
    print_num = min(30, len(test_id_list))
    print("example ids for test:", test_id_list[0:print_num])
    print("number of seqs for test:", len(test_seq_dir_list))
    print("####################################################")

    return train_source, test_source
