'''
# Base
export MODEL=Resconv_CASIA_B_rt64_train_base_bin16 && \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29505 train.py \
--dataset CASIA_B --resolution 64 --dataset_path /dev/shm/Dataset/casia_b/silhouettes_cut_pkl --pid_num 73 --batch_size 8 16 \
--milestones 10000 20000 30000 --total_iter 35000 --warmup False --dataset_augment erase_0.2 \
--backbone Resconv --sample_type fixed_unorder --channels 32 64 128 --bin_num 16 --hidden_dim 256 \
--base_norm_bin 1 --base_norm_type DDP_Sync_BN --base_norm_affine True \
--model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
--encoder_triplet_weight 1.0 --encoder_triplet_margin 0.2 --encoder_triplet_dist_type euc \
--separate_bnneck True --encoder_entropy_weight 1.0 --encoder_entropy_linear_type COSINE --encoder_entropy_linear_scale fixed_16 \
--AMP True --DDP True \
2>&1 | tee $MODEL.log
'''

# Untarget
export MODEL=RUN1_V1_CASIA_B_Untargeted_unfixed_unorder && \
python -u Adv_Gen.py \
--dataset CASIA_B --resolution 64 --dataset_path /dev/shm/Dataset/casia_b/silhouettes_cut_pkl --pid_num 73 --batch_size 8 16 \
--backbone Resconv --sample_type fixed_unorder --channels 32 64 128 --bin_num 16 --hidden_dim 256 \
--base_norm_bin 1 --base_norm_type DDP_Sync_BN --base_norm_affine True \
--test_set test --batch_size 1 --feat_idx 0 --dist_type euc --remove_no_gallery True --resume False \
--ckp_prefix adv_checkpoint/Resconv_CASIA_B_rt64_train_base_bin16-35000- --gpu 1 \
--attack_mode Untargeted --attack_name $MODEL --attack_edge_only True --attack_edge_thres 1.0 \
--attack_sample_type unfixed_unorder --attack_num_template 0.5 --attack_init_template False \
--attack_max_iter 20 --attack_optimizer_type ADAM --attack_lr 1.0 \
--attack_pushaway_weight 1.0 --attack_pushaway_margin 10.0 --attack_overlap_weight 20.0 \
2>&1 | tee $MODEL.log

export MODEL=RUN1_V2_CASIA_B_Untargeted_unfixed_unorder && \
python -u Adv_Gen.py \
--dataset CASIA_B --resolution 64 --dataset_path /dev/shm/Dataset/casia_b/silhouettes_cut_pkl --pid_num 73 --batch_size 8 16 \
--backbone Resconv --sample_type fixed_unorder --channels 32 64 128 --bin_num 16 --hidden_dim 256 \
--base_norm_bin 1 --base_norm_type DDP_Sync_BN --base_norm_affine True \
--test_set test --batch_size 1 --feat_idx 0 --dist_type euc --remove_no_gallery True --resume False \
--ckp_prefix adv_checkpoint/Resconv_CASIA_B_rt64_train_base_bin16-35000- --gpu 2 \
--attack_mode Untargeted --attack_name $MODEL --attack_edge_only True --attack_edge_thres 1.0 \
--attack_sample_type unfixed_unorder --attack_num_template 0.9 --attack_init_template False \
--attack_max_iter 20 --attack_optimizer_type ADAM --attack_lr 1.0 \
--attack_pushaway_weight 1.0 --attack_pushaway_margin 10.0 --attack_overlap_weight 20.0 \
2>&1 | tee $MODEL.log

# Target
export MODEL=RUN1_V1_CASIA_B_Targeted_unfixed_unorder && \
python -u Adv_Gen.py \
--dataset CASIA_B --resolution 64 --dataset_path /dev/shm/Dataset/casia_b/silhouettes_cut_pkl --pid_num 73 --batch_size 8 16 \
--backbone Resconv --sample_type fixed_unorder --channels 32 64 128 --bin_num 16 --hidden_dim 256 \
--base_norm_bin 1 --base_norm_type DDP_Sync_BN --base_norm_affine True \
--test_set test --batch_size 1 --feat_idx 0 --dist_type euc --remove_no_gallery True --resume False \
--ckp_prefix adv_checkpoint/Resconv_CASIA_B_rt64_train_base_bin16-35000- --gpu 3 \
--attack_mode Targeted --attack_name $MODEL --attack_edge_only True --attack_edge_thres 1.0 \
--attack_sample_type unfixed_unorder --attack_num_template 0.5 --attack_init_template False \
--attack_max_iter 20 --attack_optimizer_type ADAM --attack_lr 1.0 \
--attack_pullclose_weight 1.0 --attack_pushaway_weight 0.0 --attack_pushaway_margin 10.0 --attack_overlap_weight 20.0 \
2>&1 | tee $MODEL.log

export MODEL=RUN1_V2_CASIA_B_Targeted_unfixed_unorder && \
python -u Adv_Gen.py \
--dataset CASIA_B --resolution 64 --dataset_path /dev/shm/Dataset/casia_b/silhouettes_cut_pkl --pid_num 73 --batch_size 8 16 \
--backbone Resconv --sample_type fixed_unorder --channels 32 64 128 --bin_num 16 --hidden_dim 256 \
--base_norm_bin 1 --base_norm_type DDP_Sync_BN --base_norm_affine True \
--test_set test --batch_size 1 --feat_idx 0 --dist_type euc --remove_no_gallery True --resume False \
--ckp_prefix adv_checkpoint/Resconv_CASIA_B_rt64_train_base_bin16-35000- --gpu 4 \
--attack_mode Targeted --attack_name $MODEL --attack_edge_only True --attack_edge_thres 1.0 \
--attack_sample_type unfixed_unorder --attack_num_template 0.9 --attack_init_template False \
--attack_max_iter 20 --attack_optimizer_type ADAM --attack_lr 1.0 \
--attack_pullclose_weight 1.0 --attack_pushaway_weight 0.0 --attack_pushaway_margin 10.0 --attack_overlap_weight 20.0 \
2>&1 | tee $MODEL.log