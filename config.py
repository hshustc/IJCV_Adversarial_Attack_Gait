import argparse
import torch
import numpy as np
import random

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str, help='gpu id')
parser.add_argument('--model_name', default='GaitEncoder', type=str, help='checkpoint name for saving')
parser.add_argument('--random_seed', default=2019, type=int, help='random_seed')
#data
parser.add_argument('--dataset', default=['CASIA_B'], type=str, nargs='+', help='name of dataset')
parser.add_argument('--dataset_path', default=['/dev/shm/Dataset/casia_b/silhouettes_cut_pkl'], type=str, nargs='+', help='path to dataset')
parser.add_argument('--dataset_augment', default=None, type=str, nargs='+', help='dataset augmentation') # erase_0.5, rotate_0.5, flip_0.5, crop_0.5, persp_0.5
parser.add_argument('--check_frames', default=True, type=boolean_string, help='check minimum frames for each seq')
parser.add_argument('--resolution', default=64, type=int, help='image resolution')
parser.add_argument('--pid_json', default=None, type=str, help='json for split train and test')
parser.add_argument('--pid_num', default=73, type=int, help='split train and test')
parser.add_argument('--pid_shuffle', default=False, type=boolean_string, help='shuffle dataset or not')
parser.add_argument('--num_workers', default=48, type=int, help='workers to load data')
parser.add_argument('--frame_num_fixed', default=30, type=int, help='frames per sequence')
parser.add_argument('--frame_num_skip', default=10, type=int, help='skip frames for fixed frame_num')
parser.add_argument('--frame_num_min', default=20, type=int, help='min for unfixed frame_num')
parser.add_argument('--frame_num_max', default=40, type=int, help='max for unfixed frame_num')
parser.add_argument('--batch_size', default=[8, 16], type=int, nargs='+', help='batch size')
parser.add_argument('--sample_type', default='fixed_unorder', type=str, choices=['fixed_unorder', 'fixed_order', 'unfixed_unorder', 'all'], help='sample type')
#optimizer
parser.add_argument('--optimizer_type', default='SGD', choices=['SGD', 'ADAM'], type=str, help='SGD or ADAM')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for SGD')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--milestones', default=[10000, 20000, 30000], type=int, nargs='+', help='milestones for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for SGD')
parser.add_argument('--init_model', default=None, type=str, help='checkpoint name for initialization')
parser.add_argument('--restore_iter', default=0, type=int, help='restore iteration')
parser.add_argument('--total_iter', default=40000, type=int, help='total iteration')
parser.add_argument('--warmup', default=True, type=boolean_string, help='warm up')
parser.add_argument('--AMP', default=False, type=boolean_string, help='automatic mixed precision')
parser.add_argument('--DDP', default=False, type=boolean_string, help='distributed data parallel')
parser.add_argument('--local_rank', default=0, type=int, help='local rank for DDP')
#encoder
parser.add_argument('--backbone', default='Plain', type=str, choices=['Plain', 'Respool', 'Resconv', 'Resconv3D', 'ResconvP3D', 'ResconvAloneSA', 'ResconvEmbedSA'], help='backbone')
parser.add_argument('--channels', default=[64,128,256,512], type=int, nargs='+', help='conv channels')
parser.add_argument('--blocks', default=[1,1,1,1], type=int, nargs='+', help='conv blocks')
parser.add_argument('--strides', default=[1,2,2,1], type=int, nargs='+', help='conv strides')
parser.add_argument('--bin_num', default=[16], type=int, nargs='+', help='bin num')
parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim')
##base norm
parser.add_argument('--base_norm_bin', default=1, type=int, help='base norm bin')
parser.add_argument('--base_norm_type', default='DDP_Sync_BN', type=str, help='base norm type')
parser.add_argument('--base_norm_affine', default=True, type=boolean_string, help='base norm affine')
##part norm
parser.add_argument('--pmap_type', default='OneFC', type=str, help='part map type')
parser.add_argument('--pmap_pre_norm_type', default='None', type=str, help='part map pre norm type')
parser.add_argument('--pmap_pre_norm_affine', default=False, type=boolean_string, help='part map pre norm affine')
parser.add_argument('--pmap_mid_norm_type', default=['None'], type=str, nargs='+', help='part map mid norm type')
parser.add_argument('--pmap_mid_norm_affine', default=[False], type=boolean_string, nargs='+', help='part map mid norm affine')
parser.add_argument('--pmap_after_norm_type', default='None', type=str, help='part map after norm type')
parser.add_argument('--pmap_after_norm_affine', default=True, type=boolean_string, help='part map after norm affine')
##loss
parser.add_argument('--encoder_triplet_weight', default=1.0, type=float, help='weight for triplet after encoder') # triplet
parser.add_argument('--encoder_triplet_margin', default=0.2, type=float, help='margin for triplet after encoder') # triplet
parser.add_argument('--encoder_triplet_dist_type', default='euc', type=str, help='dist type for triplet after encoder') # triplet
parser.add_argument('--encoder_entropy_weight', default=0.0, type=float, help='weight for entropy after encoder') # entropy
parser.add_argument('--encoder_entropy_linear_drop', default=0.0, type=float, help='linear drop for entropy after encoder') # entropy
parser.add_argument('--encoder_entropy_linear_type', default='NORMAL', choices=['NORMAL', 'COSINE'], type=str, help='linear type for entropy after encoder') # entropy
parser.add_argument('--encoder_entropy_linear_scale', default='fixed_16', type=str, help='linear scale for entropy after encoder') # entropy
parser.add_argument('--encoder_entropy_label_smooth', default=False, type=boolean_string, help='label smooth for entropy after encoder') # entropy 
parser.add_argument('--encoder_supcon_weight', default=0.0, type=float, help='weight for supcon after encoder') # supcon
parser.add_argument('--encoder_supcon_temperature', default=16.0, type=float, help='temperature for supcon after encoder') # supcon
parser.add_argument('--separate_bnneck', default=False, type=boolean_string, help='separate bnneck for each part')
###################################################################################################
#test
parser.add_argument('--test_set', default='test', choices=['train', 'test'], type=str, help='train or test set for eval')
parser.add_argument('--ckp_prefix', default=None, type=str, help='ckp_prefix: prefix of the checkpoint to load')
parser.add_argument('--save_prefix', default='EVAL', type=str, help='prefix of saving')
parser.add_argument('--feat_idx', default=0, type=int, help='feat index')
parser.add_argument('--dist_type', default='euc', choices=['euc', 'cos', 'emd'], type=str, help='euclidean or cosine distance for test')
parser.add_argument('--cos_sim_thres', default=0.7, type=float, help='cosine simlarity threshold')
parser.add_argument('--rank', default=[1], type=int, nargs='+', help='rank list for show')
parser.add_argument('--max_rank', default=20, type=int, help='max rank for CMC')
parser.add_argument('--reranking', default=False, type=boolean_string, help='reranking or not')
parser.add_argument('--relambda', default=0.7, type=float, help='lambda for re-ranking')
parser.add_argument('--exclude_idt_view', default=True, type=boolean_string, help='excluding identical-view cases')
parser.add_argument('--remove_no_gallery', default=False, type=boolean_string, help='remove those thave have no gallery')
parser.add_argument('--resume', default=False, type=boolean_string, help='resume or not')
###################################################################################################
# attack
parser.add_argument('--attack_method', default='EOAA', type=str, help='attack mode')
parser.add_argument('--attack_mode', default='Untargeted', type=str, choices=['Untargeted', 'Targeted'], help='attack mode')
parser.add_argument('--attack_name', default='Base', type=str, help='attack save name')
parser.add_argument('--attack_edge_only', default=True, type=boolean_string, help='attack edge only')
parser.add_argument('--attack_edge_thres', default=1.0, type=float, help='attack edge threshold')
parser.add_argument('--attack_save_silhouette', default=False, type=boolean_string, help='attack save silhouette')
parser.add_argument('--attack_save_pickle', default=False, type=boolean_string, help='attack save pickle')
parser.add_argument('--attack_save_checkpoint', default=False, type=boolean_string, help='attack save checkpoint')
parser.add_argument('--attack_sample_type', default='unfixed_order', type=str, choices=['fixed_order', 'fixed_unorder', 'unfixed_order', 'unfixed_unorder'], help='attack sample type')
parser.add_argument('--attack_num_template', default=0.5, type=float, help='attack num template')
parser.add_argument('--attack_init_template', default=False, type=boolean_string, help='attack init template')
parser.add_argument('--attack_init_thres', default=5, type=float, help='attack init threshold')
parser.add_argument('--attack_max_iter', default=10, type=int, help='attack max iter')
parser.add_argument('--attack_optimizer_type', default='ADAM', choices=['SGD', 'ADAM', 'ADAMW'], type=str, help='attack SGD or ADAM')
parser.add_argument('--attack_lr', default=1.0, type=float, help='attack learning rate')
parser.add_argument('--attack_pushaway_weight', default=1.0, type=float, help='attack pushaway weight')
parser.add_argument('--attack_pushaway_margin', default=10.0, type=float, help='attack pushaway margin')
parser.add_argument('--attack_overlap_weight', default=20.0, type=float, help='attack overlap edge weight')
# targeted attack
parser.add_argument('--attack_num_fake_gallery', default=1, type=int, help='number of fake gallery for targeted attack')
parser.add_argument('--attack_agg_fake_gallery', default='MeanFeat', type=str, choices=['MeanFeat', 'MeanDist', 'MaxDist'], help='aggregate fake gallery for targeted attack')
parser.add_argument('--attack_pullclose_weight', default=0.0, type=float, help='attack pullclose weight')
# baseline
parser.add_argument('--attack_base_iter', default=10, type=int, help='attack base iter')
parser.add_argument('--attack_base_epsilon', default=100.0, type=float, help='attack base epsilon')
parser.add_argument('--attack_base_alpha', default=10.0, type=float, help='attack base alpha')
parser.add_argument('--attack_base_momentum', default=1.0, type=float, help='attack base momentum')
# eval
parser.add_argument('--attack_info', default=None, type=str, help='attack info')
parser.add_argument('--attack_resume', default=False, type=boolean_string, help='attack resume')
###################################################################################################
config = vars(parser.parse_args())
print('#######################################')
print("Config:", config)
print('#######################################')

SEED=config['random_seed']
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False