OUTPUT_DIR='logs/check_things_pretraining/k710distill_vits_p16x1_224_b90x1gpu_dota_mask75'
#DATA_PATH='YOUR_PATH/list_kinetics-400/train.csv'
DATA_PATH='/mnt/experiments/sorlova/datasets/DoTA_refined'
MODEL_PATH='logs/pretrained/distill/vit_s_k710_dl_from_giant.pth'

OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
    --master_port 12320 \
    run_mae_pretraining.py \
    --model pretrain_videomae_small_patch16_224 \
    --decoder_depth 4 \
    --from_ckpt ${MODEL_PATH} \
    --data_set DoTA \
    --data_path ${DATA_PATH} \
    --mask_type tube \
    --mask_ratio 0.9 \
    --lr 1.5e-4 \
    --batch_size 90 \
    --num_frames 16 \
    --sampling_rate 12 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --drop_path 0.2 \
    --warmup_epochs 2 \
    --save_ckpt_freq 1 \
    --epochs 10 \
    --num_workers 8 \
    --no_pin_mem \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}
