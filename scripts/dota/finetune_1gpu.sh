# Set the path to save checkpoints
OUTPUT_DIR='logs/dota/debug_logging/'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/mnt/experiments/sorlova/datasets/DoTA'
# path to pretrain model
MODEL_PATH='logs/pretrained/distill/vit_s_k710_dl_from_giant.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# nnodes 4, batch size 16 for 8 GPUs

# batch_size can be adjusted according to number of GPUs
# this script is for 2 GPUs (1 nodes x 2 GPUs)
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
    --master_port 12340 \
    run_frame_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set DoTA \
    --nb_classes 2 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 32 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 15 \
    --num_frames 16 \
    --sampling_rate 1 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --layer_decay 0.7 \
    --epochs 150 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --enable_deepspeed 
