# Set the path to save checkpoints
OUTPUT_DIR='logs/my_pretrain/check_kinetics/'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/scratch-nvme/ml-datasets/kinetics/k700-2020'
# path to pretrain model
MODEL_PATH='logs/pretrained/distill/vit_s_k710_dl_from_giant.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32
# srun python run_frame_finetuning.py \
torchrun --nproc_per_node=1 \
    run_mae_pretraining.py \
    --data_path ${DATA_PATH} \
    --mask_type tube \
    --mask_ratio 0.9 \
    --model pretrain_videomae_small_patch16_224 \
    --decoder_depth 4 \
    --batch_size 32 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 40 \
    --save_ckpt_freq 20 \
    --epochs 1601 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}

