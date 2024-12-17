# Set the path to save checkpoints
OUTPUT_DIR='logs/dota_fixloss/focal_1gpu/OUT_DADA2k'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA2000'
# path to pretrain model
# 'logs/dota_fixloss/focal_1gpu/checkpoint-bestap/mp_rank_00_model_states.pt'
# 'logs/auroc_behavior/crossentropy/checkpoint-3/mp_rank_00_model_states.pt'
MODEL_PATH='logs/dota_fixloss/focal_1gpu/checkpoint-bestap/mp_rank_00_model_states.pt'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# nnodes 4, batch size 16 for 8 GPUs

# batch_size can be adjusted according to number of GPUs
# this script is for 2 GPUs (1 nodes x 2 GPUs)
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
    --master_port 12320 \
    run_frame_finetuning.py \
    --eval \
    --model vit_small_patch16_224 \
    --data_set DADA2k \
    --nb_classes 2 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 256 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 15 \
    --num_frames 16 \
    --sampling_rate 1 \
    --view_fps 10 \
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
