import math
import os
import os.path as osp
import random
import sys
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import TripletSampler, DistributedTripletSampler, build_data_transforms
from .loss import DistributedLossWrapper, PartTripletLoss, all_gather
from .loss import CenterLoss, CrossEntropyLoss, SupConLoss
from .solver import WarmupMultiStepLR
from .network import GaitNeXt
from .network.sync_batchnorm import DataParallelWithCallback
from .eval import cuda_dist

class Model:
    def __init__(self, config):
        self.config = deepcopy(config)
        if self.config['DDP']:
            torch.cuda.set_device(self.config['local_rank'])
            dist.init_process_group(backend='nccl')
            self.config['encoder_entropy_weight'] *= dist.get_world_size()
            self.config['encoder_supcon_weight'] *= dist.get_world_size()
            self.config['encoder_triplet_weight'] *= dist.get_world_size()
            self.random_seed = self.config['random_seed'] + dist.get_rank()
        else:
            self.random_seed = self.config['random_seed']
        
        self.config.update({'num_id': len(self.config['train_source'].label_set)})
        self.encoder = GaitNeXt(self.config).float().cuda()
        if self.config['DDP']:
            self.encoder = DDP(self.encoder, device_ids=[self.config['local_rank']], output_device=self.config['local_rank'], find_unused_parameters=True)
        else:
            self.encoder = DataParallelWithCallback(self.encoder)
        self.build_data()
        self.build_loss()
        self.build_loss_metric()
        self.build_optimizer()

        if self.config['DDP']:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

    def build_data(self):
        # data augment
        if self.config['dataset_augment'] is not None:
            self.data_transforms = build_data_transforms(self.config['dataset_augment'], resolution=self.config['resolution'], random_seed=self.random_seed) 
        
        #triplet sampler
        if self.config['DDP']:
            self.triplet_sampler = DistributedTripletSampler(self.config['train_source'], self.config['batch_size'], random_seed=self.random_seed)
        else:
            self.triplet_sampler = TripletSampler(self.config['train_source'], self.config['batch_size'])

    def build_loss(self):
        if self.config['encoder_entropy_weight'] > 0:
            self.encoder_entropy_loss = CrossEntropyLoss(self.config['num_id'], label_smooth=self.config['encoder_entropy_label_smooth'])
            if self.config['DDP']:
                self.encoder_entropy_loss = DistributedLossWrapper(self.encoder_entropy_loss, dim=0)

        if self.config['encoder_supcon_weight'] > 0:
            self.encoder_supcon_loss = SupConLoss(self.config['encoder_supcon_temperature']).float().cuda()
            if self.config['DDP']:
                self.encoder_supcon_loss = DistributedLossWrapper(self.encoder_supcon_loss, dim=0)

        if self.config['encoder_triplet_weight'] > 0:
            self.encoder_triplet_loss = PartTripletLoss(self.config['encoder_triplet_margin'], dist_type=self.config['encoder_triplet_dist_type'], \
                                            heter_mining=False, hard_mining=False, part_mining=False).float().cuda()
            if self.config['DDP']:
                self.encoder_triplet_loss = DistributedLossWrapper(self.encoder_triplet_loss, dim=1)

    def build_loss_metric(self):
        if self.config['encoder_entropy_weight'] > 0:
            self.encoder_entropy_loss_metric = [[]]

        if self.config['encoder_supcon_weight'] > 0:
            self.encoder_supcon_loss_metric = [[]]

        if self.config['encoder_triplet_weight'] > 0:
            self.encoder_triplet_loss_metric = [[], []]

        self.total_loss_metric = []
    
    def build_optimizer(self):
        #lr and weight decay
        base_lr = self.config['lr']
        base_weight_decay = self.config['weight_decay'] if base_lr > 0 else 0

        #params
        tg_params = self.encoder.parameters() 
 
        #optimizer
        if self.config['optimizer_type'] == 'SGD':
            self.optimizer = optim.SGD(tg_params, lr=self.config['lr'], weight_decay=self.config['weight_decay'], momentum=self.config['momentum'])
        elif self.config['optimizer_type'] == 'ADAM': #if ADAM set the first stepsize equal to total_iter
            self.optimizer = optim.Adam(tg_params, lr=self.config['lr'])
        if self.config['warmup']:
            self.scheduler = WarmupMultiStepLR(self.optimizer, milestones=self.config['milestones'], gamma=self.config['gamma'])
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['milestones'], gamma=self.config['gamma'])

        #AMP
        if self.config['AMP']:
            self.scaler = GradScaler()

    def fit(self):
        self.encoder.train()
        if self.config['restore_iter'] > 0:
            self.load(self.config['restore_iter'])
        else:
            self.config['restore_iter'] = 0

        train_loader = tordata.DataLoader(
            dataset=self.config['train_source'],
            batch_sampler=self.triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        train_label_set = list(self.config['train_source'].label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, label, batch_frame in train_loader:
            self.optimizer.zero_grad()

            seq = self.np2var(seq).float()
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.asarray(target_label)).long()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            with autocast(enabled=self.config['AMP']):
                encoder_feature, encoder_bn_feature, encoder_cls_score \
                = self.encoder(seq, batch_frame, target_label)

            loss = torch.zeros(1).to(encoder_feature.device)

            if self.config['encoder_entropy_weight'] > 0:
                entropy_loss_metric = 0
                for i in range(encoder_cls_score.size(1)):
                    entropy_loss_metric += self.encoder_entropy_loss(encoder_cls_score[:, i, :].contiguous().float(), target_label)
                entropy_loss_metric = entropy_loss_metric / encoder_cls_score.size(1)
                loss += entropy_loss_metric.mean() * self.config['encoder_entropy_weight']
                self.encoder_entropy_loss_metric[0].append(entropy_loss_metric.mean().data.cpu().numpy())

            if self.config['encoder_supcon_weight'] > 0:
                supcon_loss_metric = 0
                for i in range(encoder_bn_feature.size(1)):
                    supcon_loss_metric += self.encoder_supcon_loss(encoder_bn_feature[:, i, :].contiguous().float(), target_label)
                supcon_loss_metric = supcon_loss_metric / encoder_bn_feature.size(1)
                loss += supcon_loss_metric.mean() * self.config['encoder_supcon_weight']
                self.encoder_supcon_loss_metric[0].append(supcon_loss_metric.mean().data.cpu().numpy())

            if self.config['encoder_triplet_weight'] > 0:
                encoder_triplet_feature = encoder_feature.float().permute(1, 0, 2).contiguous()
                triplet_label = target_label.unsqueeze(0).repeat(encoder_triplet_feature.size(0), 1)
                triplet_loss_metric, nonzero_num = self.encoder_triplet_loss(encoder_triplet_feature, triplet_label)
                loss += triplet_loss_metric.mean() * self.config['encoder_triplet_weight']
                self.encoder_triplet_loss_metric[0].append(triplet_loss_metric.mean().data.cpu().numpy())
                self.encoder_triplet_loss_metric[1].append(nonzero_num.mean().data.cpu().numpy())         

            self.total_loss_metric.append(loss.data.cpu().numpy())

            if loss > 1e-9:
                if self.config['AMP']:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:  
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                
            if self.config['restore_iter'] % 100 == 0:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    print(datetime.now() - _time1)
                    _time1 = datetime.now()
                    self.print_info()
                self.build_loss_metric()
            if self.config['restore_iter'] % 10000 == 0 or self.config['restore_iter'] == self.config['total_iter']:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    self.save()
            if self.config['restore_iter'] == self.config['total_iter']:
                break
            self.config['restore_iter'] += 1

    def print_info(self):
        print('iter {}:'.format(self.config['restore_iter']))

        def print_loss_info(loss_name, loss_metric, loss_weight, loss_info):
            print('{:#^30}: loss_metric={:.6f}, loss_weight={:.6f}, {}'.format(loss_name, np.mean(loss_metric), loss_weight, loss_info))

        if self.config['encoder_entropy_weight'] > 0:
            loss_name = 'Encoder Entropy'
            loss_metric = self.encoder_entropy_loss_metric[0]
            loss_weight = self.config['encoder_entropy_weight']
            loss_info = 'separate_bnneck={}, num_classes={}, linear_drop={}, linear_type={}, linear_scale={}, label_smooth={}'.format( \
                self.config['separate_bnneck'], self.config['num_id'], self.config['encoder_entropy_linear_drop'], \
                self.config['encoder_entropy_linear_type'], self.config['encoder_entropy_linear_scale'], self.config['encoder_entropy_label_smooth'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        if self.config['encoder_supcon_weight'] > 0:
            loss_name = 'Encoder Supcon'
            loss_metric = self.encoder_supcon_loss_metric[0]
            loss_weight = self.config['encoder_supcon_weight']
            loss_info = 'separate_bnneck={}, temperature={}'.format(self.config['separate_bnneck'], self.config['encoder_supcon_temperature'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        if self.config['encoder_triplet_weight'] > 0:
            loss_name = 'Encoder Triplet'
            loss_metric = self.encoder_triplet_loss_metric[0]
            loss_weight = self.config['encoder_triplet_weight']
            loss_info = 'nonzero_num={:.6f}, margin={}, dist_type={}'.format( \
                            np.mean(self.encoder_triplet_loss_metric[1]), self.config['encoder_triplet_margin'], self.config['encoder_triplet_dist_type'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        print('{:#^30}: total_loss_metric={:.6f}'.format('Total Loss', np.mean(self.total_loss_metric)))
        
        # optimizer
        # print('{:#^30}: type={}, base_lr={:.6f}, base_weight_decay={:.6f}'.format( \
        #     'Optimizer', self.config['optimizer_type'], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['weight_decay']))
        print('{:#^30}: type={}, lr={}, weight_decay={}'.format( \
            'Optimizer', self.config['optimizer_type'], \
            [params['lr'] for params in self.optimizer.param_groups], \
            [params['weight_decay'] for params in self.optimizer.param_groups]))           
        sys.stdout.flush()

    def transform(self, flag, batch_size=1, feat_idx=0):
        self.encoder.eval()
        source = self.config['test_source'] if flag == 'test' else self.config['train_source']
        self.config['sample_type'] = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        feature_list = list()
        view_list = [tmp.split('/')[-1] for tmp in source.seq_dir_list]
        seq_type_list = [tmp.split('/')[-2] for tmp in source.seq_dir_list]
        label_list = list()

        with torch.no_grad():
            for i, x in enumerate(data_loader):
                seq, label, batch_frame = x
                seq = self.np2var(seq).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()
                output = self.encoder(seq, batch_frame)
                feature = output[feat_idx]
                feature_list.append(feature.detach())             
                label_list += label

        return torch.cat(feature_list, 0), view_list, seq_type_list, label_list

    def episode_input(self, seq, episode_sampling, episode_length, episode_stride):
        assert(seq.size(0) == 1)
        num_sils = seq.shape[1]
        num_sampling = math.ceil(num_sils*1.0/episode_stride)
        input_data = []
        if episode_sampling == 'unordered':
            for i in range(num_sampling):
                if num_sils > episode_length:
                    sample_index = np.random.choice(np.arange(num_sils), episode_length, replace=False)
                else:
                    sample_index = np.arange(num_sils)
                # print('sample_index={}'.format(sample_index))
                input_data.append(seq[:, sample_index, :, :])
        elif episode_sampling == 'ordered':
            for i in range(num_sampling):
                start_idx = i * episode_length
                end_idx = (i+1) * episode_length
                if end_idx > num_sils:
                    end_idx = num_sils
                    start_idx = max(0, num_sils - episode_length)
                # print('start_idx={}, end_idx={}'.format(start_idx, end_idx))
                input_data.append(seq[:, start_idx:end_idx, :, :])
        else:
            print("Unknown Episode Sampling Type")
            os._exit(0)
        return torch.cat(input_data, 0)

    def episode_transform(self, flag, batch_size=1, feat_idx=0, episode_sampling='ordered', episode_length=30, episode_stride=30, ):
        self.encoder.eval()
        source = self.config['test_source'] if flag == 'test' else self.config['train_source']
        self.config['sample_type'] = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=1,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        feature_list = list()
        view_list = [tmp.split('/')[-1] for tmp in source.seq_dir_list]
        seq_type_list = [tmp.split('/')[-2] for tmp in source.seq_dir_list]
        label_list = list()

        with torch.no_grad():
            for i, x in enumerate(data_loader):
                seq, label, batch_frame = x
                seq = self.np2var(seq).float() # 1 x num_frames x height x width
                # if batch_frame is not None:
                #     batch_frame = self.np2var(batch_frame).int()
                # output = self.encoder(seq, batch_frame)
                if episode_length > 0:
                    # num_episode x episode_length x height x width
                    input_seq = self.episode_input(seq, episode_sampling, episode_length, episode_stride)
                else:
                    input_seq = seq
                print('sampling={}, length={}, stride={}, index={}, seq={}, input_seq={}'.format(\
                        episode_sampling, episode_length, episode_stride, i, seq.size(), input_seq.size()))
                output = self.encoder(input_seq, batch_frames=None)
                feature = output[feat_idx]
                feature_list.append(feature.detach()) # num_eposide x num_parts x hidden_dim
                label_list += label

        return feature_list, view_list, seq_type_list, label_list

    def collate_fn(self, batch):
        batch_size = len(batch)
        seqs = [batch[i][0] for i in range(batch_size)]
        label = [batch[i][1] for i in range(batch_size)]
        batch = [seqs, label, None]
        batch_frames = []
        if self.config['DDP']:
            gpu_num = 1
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
        batch_per_gpu = math.ceil(batch_size / gpu_num)

        # generate batch_frames for next step
        for gpu_id in range(gpu_num):
            batch_frames_sub = []
            for i in range(batch_per_gpu * gpu_id, batch_per_gpu * (gpu_id + 1)):
                if i < batch_size:
                    if self.config['sample_type'] == 'all':
                        batch_frames_sub.append(seqs[i].shape[0])
                    if self.config['sample_type'].split('_')[0] == 'fixed':
                        batch_frames_sub.append(self.config['frame_num_fixed'])
                    elif self.config['sample_type'].split('_')[0] == 'unfixed':
                        frame_num = np.random.randint(self.config['frame_num_min'], self.config['frame_num_max']+1)
                        batch_frames_sub.append(frame_num)
            batch_frames.append(batch_frames_sub)
        if len(batch_frames[-1]) != batch_per_gpu:
            for i in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)

        # select frames from each seq 
        def select_frame(index):
            sample = seqs[index]
            frame_set = np.arange(sample.shape[0])
            frame_num = batch_frames[int(index / batch_per_gpu)][int(index % batch_per_gpu)]
            if self.config['sample_type'] == 'all':
                frame_id_list = frame_set
            elif self.config['sample_type'].split('_')[1] == 'unorder':
                if len(frame_set) >= frame_num:
                    frame_id_list = sorted(np.random.choice(frame_set, frame_num, replace=False))
                else:
                    frame_id_list = sorted(np.random.choice(frame_set, frame_num, replace=True))
            elif self.config['sample_type'].split('_')[1] == 'order':
                seq_len = len(frame_set)
                new_frame_num = frame_num + self.config['frame_num_skip']
                if seq_len < new_frame_num:
                    frame_set = np.asarray(list(frame_set) * math.ceil(new_frame_num/seq_len))
                    seq_len = len(frame_set)
                start = np.random.randint(0, seq_len - new_frame_num + 1)
                frame_id_list = sorted(np.random.choice(frame_set[start:start+new_frame_num], frame_num, replace=False))
            return sample[frame_id_list, :, :]
        seqs = list(map(select_frame, range(len(seqs))))        

        # data augmentation
        def transform_seq(index):
            sample = seqs[index]
            return self.data_transforms(sample)
        if self.config['dataset_augment']:
            seqs = list(map(transform_seq, range(len(seqs))))  

        # concatenate seqs for each gpu if necessary
        if self.config['sample_type'] == 'fixed_unorder' or self.config['sample_type'] == 'fixed_order':
            seqs = np.asarray(seqs)                      
        elif self.config['sample_type'] == 'all' or self.config['sample_type'] == 'unfixed_unorder':
            max_sum_frames = np.max([np.sum(batch_frames[gpu_id]) for gpu_id in range(gpu_num)])
            new_seqs = []
            for gpu_id in range(gpu_num):
                tmp = []
                for i in range(batch_per_gpu * gpu_id, batch_per_gpu * (gpu_id + 1)):
                    if i < batch_size:
                        tmp.append(seqs[i])
                tmp = np.concatenate(tmp, 0)
                tmp = np.pad(tmp, \
                    ((0, max_sum_frames - tmp.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
                new_seqs.append(np.asarray(tmp))
            seqs = np.asarray(new_seqs)  

        batch[0] = seqs
        if self.config['sample_type'] == 'all' or self.config['sample_type'] == 'unfixed_unorder':
            batch[-1] = np.asarray(batch_frames)
        
        return batch

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x)) 

    def save(self):
        os.makedirs(osp.join('checkpoint', self.config['model_name']), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.config['model_name'],
                            '{}-{:0>5}-encoder.ptm'.format(self.config['save_name'], self.config['restore_iter'])))
        torch.save([self.optimizer.state_dict(), self.scheduler.state_dict()],
                   osp.join('checkpoint', self.config['model_name'],
                            '{}-{:0>5}-optimizer.ptm'.format(self.config['save_name'], self.config['restore_iter'])))

    def load(self, restore_iter):
        if self.config['DDP']:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        else:
            map_location = None
        encoder_ckp = torch.load(osp.join(
            'checkpoint', self.config['model_name'],
            '{}-{:0>5}-encoder.ptm'.format(self.config['save_name'], restore_iter)), map_location=map_location)
        self.encoder.load_state_dict(encoder_ckp)
        optimizer_ckp = torch.load(osp.join(
            'checkpoint', self.config['model_name'],
            '{}-{:0>5}-optimizer.ptm'.format(self.config['save_name'], restore_iter)), map_location=map_location)
        self.optimizer.load_state_dict(optimizer_ckp[0])
        self.scheduler.load_state_dict(optimizer_ckp[1])  

    def init_model(self, init_model):
        if self.config['DDP']:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        else:
            map_location = None
        encoder_state_dict = self.encoder.state_dict()
        ckp_state_dict = torch.load(init_model, map_location=map_location)
        init_state_dict = {k: v for k, v in ckp_state_dict.items() if k in encoder_state_dict}
        drop_state_dict = {k: v for k, v in ckp_state_dict.items() if k not in encoder_state_dict}
        print('#######################################')
        if init_state_dict:
            print("Useful Layers in Init_model for Initializaiton:\n", init_state_dict.keys())
        else:
            print("None of Layers in Init_model is Used for Initializaiton.")
        print('#######################################')
        if drop_state_dict:
            print("Useless Layers in Init_model for Initializaiton:\n", drop_state_dict.keys())
        else:
            print("All Layers in Init_model are Used for Initialization.")
        encoder_state_dict.update(init_state_dict)
        none_init_state_dict = {k: v for k, v in encoder_state_dict.items() if k not in init_state_dict}
        print('#######################################')
        if none_init_state_dict:
            print("The Layers in Target_model that Are *Not* Initialized:\n", none_init_state_dict.keys())
        else:
            print("All Layers in Target_model are Initialized")  
        print('#######################################')      
        self.encoder.load_state_dict(encoder_state_dict)    
