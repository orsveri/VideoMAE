# Set the path to save checkpoints
OUTPUT_DIR='logs/check_things/after_freeze_finetune_frombest'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/mnt/experiments/sorlova/datasets/DoTA_refined'
# path to pretrain model
#'logs/pretrained/distill/vit_s_k710_dl_from_giant.pth'
MODEL_PATH='logs/check_things/after_freeze_finetune/checkpoint-bestap/mp_rank_00_model_states.pt'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
    --master_port 12320 \
    run_frame_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set DoTA \
    --loss crossentropy \
    --nb_classes 2 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 64 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --warmup_epochs 10 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --layer_decay 0.7 \
    --epochs 20 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --enable_deepspeed 
