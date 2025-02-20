#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=14
#SBATCH --time=80:00:00
#SBATCH --output=jobs/job_outputs/pretrain_k700_bl1_decord_%j.out

# For H100 nodes:
#export NCCL_SOCKET_IFNAME="eno2np0"
#export NCCL_DEBUG=INFO
#export CUDA_LAUNCH_BLOCKING=1

module load 2023
module load Anaconda3/2023.07-2

export OMP_NUM_THREADS=16
export MASTER_PORT=12345
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
OUTPUT_DIR='logs/my_pretrain/bl1_vits_k700/'
# path to Kinetics set (train.csv/val.csv/test.csv
DATA_PATH='/scratch-nvme/ml-datasets/kinetics/k700-2020'
# path to pretrain model
MODEL_PATH='logs/pretrained/k400_vits/videomae_vits_k400_pretrain_ckpt.pth'

# For BDD and BDD+CAP-DATA pretraining, we use: 
#   - 1M samples per epoch    -> 1250 steps per epoch (over 4 GPUs in total) -> 20 epochs to achieve 25K steps
# With Kinetics, we have only ~0.5M videos and we sample one video once per epoch be default, so:
#   - 536685 samples per epoch -> 670 steps per epoch (over 4 GPUs in total) -> 38 ep to achieve 25K steps
# Then we stopped training after 12 epochs, and here it will be 22-23 epochs
torchrun --nproc_per_node=4 \
    run_mae_pretraining.py \
    --data_path ${DATA_PATH} \
    --mask_type tube \
    --mask_ratio 0.9 \
    --model pretrain_videomae_small_patch16_224 \
    --from_ckpt ${MODEL_PATH} \
    --decoder_depth 4 \
    --batch_size 200 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --epochs 38 \
    --save_ckpt_freq 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lr 3e-4 \
    --min_lr 3e-5 \
