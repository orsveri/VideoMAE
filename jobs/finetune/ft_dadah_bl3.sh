#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=18
#SBATCH --time=40:00:00
#SBATCH --output=jobs/job_outputs/ft_BL3_dadah_%j.out

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
OUTPUT_DIR='/home/sorlova/repos/AITHENA/NewStage/VideoMAE/logs/baselines/bl3/dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3'
# path to data set 
DATA_PATH="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/DADA2K"
# path to pretrain model
MODEL_PATH='logs/pretrained/InternVideo/pretr_s14_single_dist1B/IntVid2_s14_single_dist1B.bin'

#     --bf16 \
torchrun --nproc_per_node=1 \
    iv2_sm_run_frame_finetuning.py \
    --model internvideo2_small_patch14_224 \
    --data_set DADA2K_half \
    --nb_classes 2 \
    --tubelet_size 1 \
    --no_use_decord \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --steps_per_print 10 \
    --batch_size 28 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
    --num_frames 8 \
    --view_fps 5 \
    --sampling_rate 1 \
    --sampling_rate_val 3 \
    --nb_samples_per_epoch 50000 \
    --num_workers 16 \
    --warmup_epochs 5 \
    --epochs 50 \
    --lr 1e-3 \
    --drop_path 0.1 \
    --head_drop_path 0.1 \
    --fc_drop_rate 0.0 \
    --layer_decay 0.75 \
    --layer_scale_init_value 1e-5 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --enable_deepspeed \
    --zero_stage 0 \
