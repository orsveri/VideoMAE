#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=14
#SBATCH --time=01:20:00
#SBATCH --output=jobs_outs/212-11_%j.out

# For H100 nodes:
export NCCL_SOCKET_IFNAME="eno2np0"
#export NCCL_DEBUG=INFO
#export CUDA_LAUNCH_BLOCKING=1

module load 2023
module load Anaconda3/2023.07-2

export OMP_NUM_THREADS=16
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
OUTPUT_DIR='logs/my_pretrain_ft_dota/bl1_vits_k400_vidmae/oldaug_videomae_vits_k400_1/ckpt-3-dada-res'
# path to data set 
DATA_PATH='/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/DADA2K'
# path to pretrain model
MODEL_PATH='logs/my_pretrain_ft_dota/bl1_vits_k400_vidmae/oldaug_videomae_vits_k400_1/checkpoint-3.pth'

#export MASTER_PORT=21567
#export MASTER_ADDR=$(hostname)

# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32

# 2 hrs per epoch
torchrun --nproc_per_node=1 \
    iv2_sm_run_frame_finetuning.py \
    --eval \
    --eval_option 212_11 \
    --model internvideo2_small_patch14_224 \
    --data_set DADA2K \
    --loss crossentropy \
    --nb_classes 2 \
    --tubelet_size 1 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 560 \
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
    --enable_deepspeed \
    --seed 424242
