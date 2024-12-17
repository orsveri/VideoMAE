# Set the path to save checkpoints
OUTPUT_DIR='logs/hmdb51/from_k400_vits_closer_settings2/'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/mnt/experiments/sorlova/datasets/HMDB51'
# path to pretrain model
MODEL_PATH='logs/pretrained/k400_vits/checkpoint.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# nnodes 4, batch size 16 for 8 GPUs

# batch_size can be adjusted according to number of GPUs
# this script is for 2 GPUs (1 nodes x 2 GPUs)
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
    --master_port 12320 \
    run_class_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set HMDB51 \
    --nb_classes 51 \
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
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --layer_decay 0.7 \
    --epochs 150 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed 
