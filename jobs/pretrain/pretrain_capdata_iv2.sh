#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=80:00:00
#SBATCH --output=jobs/job_outputs/pretrain_IV2_bdd-capdata_%j.out

# For H100 nodes:
export NCCL_SOCKET_IFNAME="eno2np0"
#export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

module load 2023
module load Anaconda3/2023.07-2

export OMP_NUM_THREADS=15
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
OUTPUT_DIR='logs/my_pretrain/bl3_iv2s/bdd-capdata_lightcrop_b150x4_mask075'
# path to data set 
DATA_PATH1='/scratch-nvme/ml-datasets/bdd100k/videos'
DATA_PATH2="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA"
# path to pretrain model
MODEL_PATH='logs/pretrained/InternVideo/pretr_s14_single_dist1B/IntVid2_s14_single_dist1B.bin'

# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32
# srun python run_frame_finetuning.py \
torchrun --nproc_per_node=4 \
    iv2_run_mae_double_pretraining.py \
    --data_set1 BDD100K \
    --data_path1 ${DATA_PATH1} \
    --sampling_rate1 16 \
    --data_set2 CAP-DATA \
    --data_path2 ${DATA_PATH2} \
    --sampling_rate2 1 \
    --mask_type tube \
    --mask_ratio 0.75 \
    --tubelet_size 1 \
    --model pretrain_videomae_internvideo2_patch14_224 \
    --from_ckpt ${MODEL_PATH} \
    --decoder_depth 4 \
    --batch_size1 90 \
    --batch_size2 60 \
    --num_frames 8 \
    --view_fps 5 \
    --transforms_finetune_align \
    --nb_samples_per_epoch 750000 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --epochs 20 \
    --save_ckpt_freq 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lr 3e-4 \
    --min_lr 3e-5 \
    --drop_path 0.1 \
