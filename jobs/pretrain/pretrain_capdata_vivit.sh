#!/bin/bash

#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=80:00:00
#SBATCH --output=jobs/job_outputs/pretrain_VIVIT_bdd-capdata_%j.out

# For H100 nodes:
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="eno2np0"

export CUDA_HOME=/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1/
#export CUDA_LAUNCH_BLOCKING=1

module load 2023
module load Anaconda3/2023.07-2

export OMP_NUM_THREADS=15

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
OUTPUT_DIR='logs/my_pretrain/bl4_vivit_bdd-capdata_lightcrop_b200x4_mask075'
# path to data set 
DATA_PATH1='/scratch-nvme/ml-datasets/bdd100k/videos'
DATA_PATH2="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA"
# path to pretrain model
# 'logs/pretrained/k400_vits/videomae_vits_k400_pretrain_ckpt.pth'
MODEL_PATH='logs/pretrained/vivit/vivit-b-16x2-kinetics400_vidmae.pth'



# torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE \
#     --rdzv_id=$SLURM_JOB_ID \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR \


# srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
#     --export=ALL \

export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 hostname --ip-address)
export MASTER_PORT=29500  # Use a high, non-conflicting port

# export RANK=${SLURM_PROCID}
# export WORLD_SIZE=${SLURM_NTASKS}
# export LOCAL_RANK=${SLURM_LOCALID}

echo "RANK=$RANK WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$LOCAL_RANK NTASKS=$SLURM_NTASKS"

# Normally, we use ViT-S based models with 1-GPU (H100) batch size 200.
# This option uses ViT-B model that requires 2x memory, so the base batch size is 100.
# So, to keep the original settings - total batch size 800, we will use 2 nodes instead of 1
srun python run_mae_double_pretraining.py \
    --data_set1 BDD100K \
    --data_path1 ${DATA_PATH1} \
    --sampling_rate1 16 \
    --data_set2 CAP-DATA \
    --data_path2 ${DATA_PATH2} \
    --sampling_rate2 1 \
    --mask_type tube \
    --mask_ratio 0.75 \
    --tubelet_size 2 \
    --model pretrain_videomae_base_patch16_224 \
    --from_ckpt ${MODEL_PATH} \
    --decoder_depth 4 \
    --batch_size1 60 \
    --batch_size2 40 \
    --num_frames 16 \
    --transforms_finetune_align \
    --nb_samples_per_epoch 1000000 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --epochs 20 \
    --save_ckpt_freq 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lr 3e-4 \
    --min_lr 3e-5 \
