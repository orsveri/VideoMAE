#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --time=50:00:00
#SBATCH --output=jobs_outs_ftpt/205_bl2_dota_%j.out

# For H100 nodes:
#export NCCL_SOCKET_IFNAME="eno2np0"
#export NCCL_DEBUG=INFO
#export CUDA_LAUNCH_BLOCKING=1

module load 2023
module load Anaconda3/2023.07-2

export OMP_NUM_THREADS=15
export MASTER_PORT=12305
export MASTER_ADDR=$(hostname)
export CUDA_HOME=/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1/

__conda_setup="$('/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda/etc/profile.d/conda.sh" ]; then
        . "/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda/etc/profile.d/conda.sh"
    else
        export PATH="/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate /home/sorlova/anaconda3/envs/video

cd /home/sorlova/repos/AITHENA/NewStage/VideoMAE

# Set the path to save checkpoints
OUTPUT_DIR='/home/sorlova/repos/AITHENA/NewStage/VideoMAE/logs/ft_after_pretrain/bl2/dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3'
# path to data set 
DATA_PATH='/gpfs/work3/0/tese0625/RiskNetData/DoTA_refined'
# path to pretrain model
# 'logs/pretrained/VideoMAE2_distill/videomae_vits_k710_distill_from_giant.pth'
MODEL_PATH='/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/my_pretrain/bl2_vits/bdd-capdata_lightcrop_b200x4_mask075/checkpoint-11.pth'


# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32

# 2 hrs per epoch
torchrun --nproc_per_node=1 \
    run_frame_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set DoTA \
    --loss crossentropy \
    --nb_classes 2 \
    --tubelet_size 2 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 56 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
    --num_frames 16 \
    --sampling_rate 1 \
    --sampling_rate_val 2 \
    --nb_samples_per_epoch 50000 \
    --num_workers 12 \
    --opt adamw \
    --lr 1e-3 \
    --min_lr 1e-6 \
    --warmup_lr 1e-6 \
    --warmup_epochs 5 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --layer_decay 0.6 \
    --aa rand-m6-n3-mstd0.5-inc1 \
    --epochs 50 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --enable_deepspeed \
    --seed 42
