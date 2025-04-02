import torch
import torch.utils.data as tordata
import torch.distributed as dist
import math
import random
import numpy as np

class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = list()
            pid_list = np.random.choice(self.dataset.label_set, self.batch_size[0], replace=False)
            for pid in pid_list:
                # _index = self.dataset.index_dict[pid]
                type_keys = list(self.dataset.index_dict[pid].keys())
                _index = list()
                for i, _type in enumerate(type_keys):
                    _type_index = self.dataset.index_dict[pid][_type]
                    _index += _type_index
                if len(_index) >= self.batch_size[1]:
                    _index = np.random.choice(_index, self.batch_size[1], replace=False).tolist()
                else:
                    _index = np.random.choice(_index, self.batch_size[1], replace=True).tolist()             
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size

def random_sample_index(obj_list, num):
    if len(obj_list) < num:
        idx = random.choices(range(len(obj_list)), k=num)
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(obj_list))[:num]
    if torch.cuda.is_available():
        idx = idx.cuda()
    return idx

def sync_random_sample_list(obj_list, k):
    idx = random_sample_index(obj_list, k)
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]

class DistributedTripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, world_size=None, rank=None, random_seed=2019):
        np.random.seed(random_seed)
        random.seed(random_seed)
        print("random_seed={} for DistributedTripletSampler".format(random_seed))
        
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        assert(self.batch_size[0] % self.world_size == 0)

    def __iter__(self):
        while (True):
            sample_indices = list()
            pid_list = sync_random_sample_list(self.dataset.label_set, self.batch_size[0])
            pid_list = pid_list[self.rank:self.batch_size[0]:self.world_size]
            for pid in pid_list:
                type_keys = list(self.dataset.index_dict[pid].keys())
                _index = list()
                for i, _type in enumerate(type_keys):
                    _type_index = self.dataset.index_dict[pid][_type]
                    _index += _type_index
                if len(_index) >= self.batch_size[1]:
                    _index = np.random.choice(_index, self.batch_size[1], replace=False).tolist()
                else:
                    _index = np.random.choice(_index, self.batch_size[1], replace=True).tolist()
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size
