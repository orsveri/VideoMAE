# Set the path to save checkpoints
OUTPUT_DIR='logs/clean_datasets/DADA2K/check_plots'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA2000'
# path to pretrain model
MODEL_PATH='logs/pretrained/distill/vit_s_k710_dl_from_giant.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# nproc_per_node is the number of used GPUs
# batch_size is set for one GPU
# batch_size=16, nproc_per_node=2 => the effective batch_size is 32
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
    --master_port 12310 \
    run_frame_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set DADA2k \
    --loss crossentropy \
    --nb_classes 2 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 32 \
    --num_sample 1 \
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
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --enable_deepspeed 
