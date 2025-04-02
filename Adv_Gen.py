import os
import os.path as osp
import numpy as np
import pickle
import shutil

from model.initialization import initialization
from config import *
from Adv_Utils import *
from Adv_Core import *

# init model
m = initialization(config)
if config['ckp_prefix'] is not None:
    print('Loading the model of %s' % config['ckp_prefix'])
    m.init_model('{}encoder.ptm'.format(config['ckp_prefix']))
print('#######################################')
print("Network Structures:", m.encoder)
print('#######################################')

# init attack
os.makedirs('adv_data', exist_ok=True)
os.makedirs('adv_basic', exist_ok=True)
os.makedirs('adv_full', exist_ok=True)

# split dataset
source = config['test_source'] if config['test_set'] == 'test' else config['train_source']
probe_path_list, gallery_path_list = split_dataset(source, config['dataset'][0])
print('total={}, num_probe={}, num_gallery={}'.format(source.data_size, len(probe_path_list), len(gallery_path_list)))

# init attack info
basic_attack_info_pkl = osp.join('adv_basic', '{}_{}set_{}_Basic_Attack_Info.pkl'.format(config['dataset'][0], config['test_set'], config['attack_mode']))
if osp.exists(basic_attack_info_pkl):
    print("{} EXISTS".format(basic_attack_info_pkl))
    with open(basic_attack_info_pkl, 'rb') as f:
        attack_info = pickle.load(f)
else:
    if config['attack_mode'] == 'Untargeted':
        attack_info = make_basic_attack_info(source, probe_path_list, gallery_path_list, targeted_attack=False, cross_view_eval=False)
    elif config['attack_mode'] == 'Targeted':
        dataname = config['dataset'][0]
        if dataname in ['CASIA_B', 'OUMVLP']:
            attack_info = make_basic_attack_info(source, probe_path_list, gallery_path_list, targeted_attack=True, cross_view_eval=True)
        elif dataname in ['Gait3D', 'GREW']:
            attack_info = make_basic_attack_info(source, probe_path_list, gallery_path_list, targeted_attack=True, cross_view_eval=False)       
    with open(basic_attack_info_pkl, 'wb') as f:
        pickle.dump(attack_info, f)
    print("{} SAVED".format(basic_attack_info_pkl))

# full attack info
full_attack_info_pkl =  osp.join('adv_full', '{}_{}set_{}_{}_Full_Attack_Info.pkl'.format(config['dataset'][0], config['test_set'], config['attack_mode'], config['attack_name']))
# root dir to save silhouette
des_sil_root_dir = osp.join('adv_data', 'SIL_{}_{}set_{}_{}'.format(config['dataset'][0], config['test_set'], config['attack_mode'], config['attack_name']))
des_pkl_root_dir = osp.join('adv_data', 'PKL_{}_{}set_{}_{}'.format(config['dataset'][0], config['test_set'], config['attack_mode'], config['attack_name']))
if config['attack_resume'] and osp.exists(full_attack_info_pkl):
    print("{} EXISTS".format(full_attack_info_pkl))
    with open(full_attack_info_pkl, 'rb') as f:
        attack_info = pickle.load(f)
else:
    for probe_idx, probe_key in enumerate(attack_info.keys()):
        print("########################################################################################################")
        print('probe_key={}, probe_total={}, probe_idx={}'.format(probe_key, len(probe_path_list), probe_idx))
        sub_attack_info = attack_info[probe_key]
        # load
        seq_path = sub_attack_info['seq_path']
        src_seq = source.__loader__(seq_path)
        # attack
        sub_attack_info.update({'attack_index':[]})
        sub_attack_info.update({'attack_seq':[]})
        sub_attack_info.update({'attack_feat':[]})
        if config['attack_mode'] == 'Untargeted':
            if config['attack_method'] == 'EOAA':
                attack_index, attack_seq, attack_feat = EOAA_Attack_Gait(m.encoder, src_seq, config)
            elif config['attack_method'] in ['FGSM','IFGSM','MIFGSM']:
                attack_index, attack_seq, attack_feat = FGSM_Attack_Gait(m.encoder, src_seq, config)
            sub_attack_info['attack_index'].append(attack_index)
            sub_attack_info['attack_seq'].append(attack_seq)
            sub_attack_info['attack_feat'].append(attack_feat)
            print('probe_key={}, fake_view={}, fake_id={}, fake_seq_path={}'.format(probe_key, \
                    sub_attack_info['fake_view'], sub_attack_info['fake_id'], sub_attack_info['fake_seq_path']))
            print('probe_key={}, attack_index={}, attack_seq={}, attack_feat={}'.format(probe_key, \
                    attack_index, attack_seq.shape, attack_feat.shape))
            #############################################################
            if config['attack_save_silhouette']:
                des_seq_path = osp.join(des_sil_root_dir, make_path_id(seq_path), make_path_type(seq_path), make_path_view(seq_path))
                save_sil(source.__recover__(src_seq), osp.join(des_seq_path, 'online_real'), attack_index=None)
                save_sil(source.__recover__(attack_seq), osp.join(des_seq_path, 'online_attack'), attack_index=attack_index)
            #############################################################
        elif config['attack_mode'] == 'Targeted':
            for i, fake_view in enumerate(sub_attack_info['fake_view']):
                fake_id = sub_attack_info['fake_id'][i]
                fake_seq_path = sub_attack_info['fake_seq_path'][i]
                fake_seq = [source.__loader__(path) for path in fake_seq_path[:config['attack_num_fake_gallery']]]
                if config['attack_method'] == 'EOAA':
                    attack_index, attack_seq, attack_feat = EOAA_Attack_Gait(m.encoder, src_seq, config, fake_seq=fake_seq)
                elif config['attack_method'] in ['FGSM','IFGSM','MIFGSM']:
                    attack_index, attack_seq, attack_feat = FGSM_Attack_Gait(m.encoder, src_seq, config, fake_seq=fake_seq)
                sub_attack_info['attack_index'].append(attack_index)
                sub_attack_info['attack_seq'].append(attack_seq)
                sub_attack_info['attack_feat'].append(attack_feat)
                print('probe_key={}, fake_view={}, fake_id={}, fake_seq_path={}'.format(probe_key, \
                        sub_attack_info['fake_view'][i], sub_attack_info['fake_id'][i], sub_attack_info['fake_seq_path'][i]))
                print('probe_key={}, attack_index={}, attack_seq={}, attack_feat={}'.format(probe_key, \
                        attack_index, attack_seq.shape, attack_feat.shape))
                #############################################################
                if config['attack_save_silhouette']:
                    des_seq_path = osp.join(des_sil_root_dir, make_path_id(seq_path), make_path_type(seq_path), make_path_view(seq_path))
                    save_sil(source.__recover__(src_seq), osp.join(des_seq_path, 'online_real'), attack_index=None)
                    save_sil(source.__recover__(attack_seq), osp.join(des_seq_path, 'online_attack_fakeview_{}_fakeid_{}'.format(fake_view, fake_id)), attack_index=attack_index)
                #############################################################
        attack_info.update({probe_key:sub_attack_info})
        print("########################################################################################################")
    with open(full_attack_info_pkl, 'wb') as f:
        pickle.dump(attack_info, f, protocol=4)
        print("{} SAVED".format(full_attack_info_pkl))

# save silhouette for visualization, and attacked test sequences are saved
if config['attack_save_silhouette']:
    for probe_key in attack_info.keys():
        sub_attack_info = attack_info[probe_key]
        seq_path = sub_attack_info['seq_path']
        des_seq_path = osp.join(des_sil_root_dir, make_path_id(seq_path), make_path_type(seq_path), make_path_view(seq_path))
        # src_seq
        src_seq = source.__loader__(seq_path)
        src_seq = source.__recover__(src_seq)
        save_sil(src_seq, osp.join(des_seq_path, 'real'))
        # attack_seq
        if config['attack_mode'] == 'Untargeted':
            attack_seq = sub_attack_info['attack_seq'][0]
            attack_seq = source.__recover__(attack_seq)
            attack_index = sub_attack_info['attack_index'][0]
            save_sil(attack_seq, osp.join(des_seq_path, 'attack'), attack_index=attack_index)
        elif config['attack_mode'] == 'Targeted':
            for i, fake_view in enumerate(sub_attack_info['fake_view']):
                fake_id = sub_attack_info['fake_id'][i]
                attack_seq = sub_attack_info['attack_seq'][i]
                attack_seq = source.__recover__(attack_seq)
                attack_index = sub_attack_info['attack_index'][i]
                save_sil(attack_seq, osp.join(des_seq_path, 'attack_fakeview_{}_fakeid_{}'.format(fake_view, fake_id)), attack_index=attack_index)
    print("{} SAVED".format(des_sil_root_dir))

# save checkpoint for debug
if config['attack_save_checkpoint']:
    torch.save(m.encoder.state_dict(), '{}_{}_{}encoder.ptm'.format(config['attack_mode'], config['attack_name'], osp.basename(config['ckp_prefix'])))
    torch.save([m.optimizer.state_dict(), m.scheduler.state_dict()], '{}_{}_{}optimizer.ptm'.format(config['attack_mode'], config['attack_name'], osp.basename(config['ckp_prefix'])))

# save pickle for evaluating other models, and full test sequences are saved
if config['attack_save_pickle']:
    for idx, seq_path in enumerate(source.seq_dir_list):
        seq_key = make_path_key(seq_path)
        if seq_key in attack_info.keys():
            sub_attack_info = attack_info[seq_key]
            assert(seq_path == sub_attack_info['seq_path'])
            des_seq_path = osp.join(des_pkl_root_dir, make_path_id(seq_path), make_path_type(seq_path))
            real_view = make_path_view(seq_path)
            # attack_seq
            if config['attack_mode'] == 'Untargeted':
                attack_seq = sub_attack_info['attack_seq'][0]
                attack_seq = source.__recover__(attack_seq, uint8=True)
                attack_index = sub_attack_info['attack_index'][0]
                save_pkl(attack_seq, osp.join(des_seq_path, '{}_attack'.format(real_view)), attack_index=attack_index)
            elif config['attack_mode'] == 'Targeted':
                for i, fake_view in enumerate(sub_attack_info['fake_view']):
                    fake_id = sub_attack_info['fake_id'][i]
                    attack_seq = sub_attack_info['attack_seq'][i]
                    attack_seq = source.__recover__(attack_seq, uint8=True)
                    attack_index = sub_attack_info['attack_index'][i]
                    save_pkl(attack_seq, osp.join(des_seq_path, '{}_attack_fakeview_{}_fakeid_{}'.format(real_view, fake_view, fake_id)), attack_index=attack_index)
        else:
            des_seq_path = osp.join(des_pkl_root_dir, make_path_id(seq_path), make_path_type(seq_path), make_path_view(seq_path)) 
            shutil.copytree(seq_path, des_seq_path)
    print("{} SAVED".format(des_pkl_root_dir))