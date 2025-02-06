#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=14
#SBATCH --time=00:30:00
#SBATCH --output=jobs/jobs_outs/eval_dota_exp8_5_%j.out

# For H100 nodes:
#export NCCL_SOCKET_IFNAME="eno2np0"
#export NCCL_DEBUG=INFO
#export CUDA_LAUNCH_BLOCKING=1

module load 2023
module load Anaconda3/2023.07-2

export OMP_NUM_THREADS=10
export MASTER_PORT=12543
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
#OUTPUT_DIR='logs/my_pretrain_ft_dota/bdd100k_extra_pretrain_vits/from-k400_full_regular_b200x4_mask075-session2_newaug'
OUTPUT_DIR='logs/my_pretrain_ft_dota/bdd100k_extra_pretrain_vits/from-k400_full_regular_b200x4_mask075-session2_newaug/ckpt-4-dota-res'
# path to data set 
DATA_PATH='/gpfs/work3/0/tese0625/RiskNetData/DoTA_refined'
# path to pretrain model
MODEL_PATH='logs/my_pretrain_ft_dota/bdd100k_extra_pretrain_vits/from-k400_full_regular_b200x4_mask075-session2_newaug/checkpoint-4.pth'


# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32

# 2 hrs per epoch
torchrun --nproc_per_node=1 \
    run_frame_finetuning.py \
    --eval \
    --eval_option 9_5 \
    --model vit_small_patch16_224 \
    --data_set DoTA \
    --loss crossentropy \
    --nb_classes 2 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 550 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
    --num_frames 16 \
    --sampling_rate 1 \
    --view_fps 10 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --epochs 20 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --enable_deepspeed
