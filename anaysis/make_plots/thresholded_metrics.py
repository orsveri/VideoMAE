import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


DoTA_dirs = [
    # MOVAD
    "/gpfs/work3/0/tese0625/VideoMAE_results/comparison/MOVAD/ft_DoTA_refined_ckpt_696/eval_DoTA_ckpt696",
    "/gpfs/work3/0/tese0625/VideoMAE_results/comparison/MOVAD/ft_DADA2K_ep702/eval_DoTA_ckpt_702",
    # CycleCrash models
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s5_vidnext_dota_lr5e6_/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s6_vidnext_dada_lr2e6_/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s7_convnext_dota_lr5e6_/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s8_convnext_dada_lr2e6_/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s9_resnetnst_dota/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s10_resnetnst_dada/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s11_r2p1d_dota/eval_DoTA_ckpt_7",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s11h_r2p1d_dotah/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s12_r2p1d_dada/eval_DoTA_ckpt_5",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s12h_r2p1d_dadah/eval_DoTA_ckpt_bestap",
    # 12-... Direct finetuning of ViFM
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/1_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_16",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/2_dotaH_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/3_dada_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/4_dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_9",
    # 16
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/5_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_5",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/6_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_8",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/7_dada_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_4",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/8_dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_5",
    # 20
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/9_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_5",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/10_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_6",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/11_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_3",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/12_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_1",
    # 24
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/13_VITB_dota_lr5e4_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_9",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/14_VITB_dotah_lr5e4_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/15_VITB_dada_lr5e4_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_6",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/16_VITB_dadah_lr5e_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_bestap",
    # 28 "Ablations"
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C1_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_25",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C2_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C3_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C4_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_bestap",
    # 32
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/101_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/102_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/103_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_8",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/104_dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_8",
    #  36 Finetuning after SSPPT
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/201_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_11",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/202_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_13",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/203_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_8",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/204_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_13",
    # 40
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/205_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_16",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/206_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/207_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/208_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_8",
    # 44
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/209_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_25",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/210_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/211_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_11",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/212_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_11",
    # 48
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl4/213_VITB_dota_lr5e4_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_18",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl4/214_VITB_dotah_lr5e4_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl4/215_VITB_dada_lr5e_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_10",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl4/216_VITB_dadah_lr5e_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_bestap"
]

DADA2K_dirs = [
    # MOVAD
    "/gpfs/work3/0/tese0625/VideoMAE_results/comparison/MOVAD/ft_DoTA_refined_ckpt_696/eval_DADA2K_ckpt_696",
    "/gpfs/work3/0/tese0625/VideoMAE_results/comparison/MOVAD/ft_DADA2K_ep702/eval_DADA2K_ckpt_702",
    # CycleCrash models
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s5_vidnext_dota_lr5e6_/eval_DADA2K_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s6_vidnext_dada_lr2e6_/eval_DADA2K_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s7_convnext_dota_lr5e6_/eval_DADA2K_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s8_convnext_dada_lr2e6_/eval_DADA2K_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s9_resnetnst_dota/eval_DADA2K_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s10_resnetnst_dada/eval_DADA2K_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s11_r2p1d_dota/eval_DADA2K_ckpt_7",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s11h_r2p1d_dotah/eval_DADA2K_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s12_r2p1d_dada/eval_DADA2K_ckpt_5",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/comparison_cc/s12h_r2p1d_dadah/eval_DADA2K_ckpt_bestap",
    # Direct finetuning of ViFM
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/1_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_16",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/2_dotaH_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/3_dada_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/4_dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_9",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/5_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_5",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/6_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_8",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/7_dada_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_4",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/8_dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_5",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/9_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_5",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/10_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_6",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/11_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_3",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/12_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_1",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/13_VITB_dota_lr5e4_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_9",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/14_VITB_dotah_lr5e4_b28x2_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/15_VITB_dada_lr5e4_b56x1_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_6",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/16_VITB_dadah_lr5e_b28x2_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_bestap",
    # "Ablations"
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C1_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_25",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C2_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C3_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C4_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_bestap",
    # 
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/101_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/102_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/103_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_8",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/104_dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_8",
    # Finetuning after SSPPT
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/201_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_11",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/202_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_13",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/203_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_8",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/204_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_13",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/205_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_16",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/206_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/207_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/208_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_8",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/209_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_25",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/210_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/211_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_11",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/212_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_11",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl4/213_VITB_dota_lr5e4_b56x1_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_18",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl4/214_VITB_dotah_lr5e4_b28x2_dsampl1val2_ld06_aam6n3/eval_DADA2K_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl4/215_VITB_dada_lr5e_b56x1_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_10",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl4/216_VITB_dadah_lr5e_b28x2_dsampl1val3_ld06_aam6n3/eval_DADA2K_ckpt_bestap"
]



def extract_threshold(col_name, prefix):
    """Helper function to extract the threshold value from a column name.
       For example, extract_threshold("MCC_0.25", "MCC_") returns 0.25."""
    return float(col_name.replace(prefix, ""))


def plot_mcc_p_r_curves(file_paths, file_labels):
    # Prepare lists to store curves for each metric from each file.
    mcc_curves = []       # each entry: (thresholds, MCC values)
    precision_curves = [] # each entry: (thresholds, Precision values)
    recall_curves = []    # each entry: (thresholds, Recall values)

    # Loop through each file, reading data and extracting the curves from row "all_samples"
    for path in file_paths:
        df = pd.read_csv(os.path.join(path, "thresh_stats.csv"))  # assume the first column is an index (e.g. sample names)
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        # Set the "group" column as the index
        df = df.set_index("group")
        # Extract the row labeled "all_samples"
        row = df.loc["all_samples"]
        
        # Identify columns for each metric based on the prefix.
        mcc_cols = [col for col in row.index if col.startswith("mcc_")]
        prec_cols = [col for col in row.index if col.startswith("p_")]
        rec_cols = [col for col in row.index if col.startswith("r_")]
        
        # Sort the column names by the threshold value
        mcc_cols = sorted(mcc_cols, key=lambda x: extract_threshold(x, "mcc_"))
        prec_cols = sorted(prec_cols, key=lambda x: extract_threshold(x, "p_"))
        rec_cols = sorted(rec_cols, key=lambda x: extract_threshold(x, "r_"))
        
        # Extract thresholds and corresponding values as numpy arrays
        mcc_thresholds = np.array([extract_threshold(col, "mcc_") for col in mcc_cols])
        mcc_values = row[mcc_cols].values.astype(float)
        #print("MCC thresholds: ", mcc_thresholds)
        
        prec_thresholds = np.array([extract_threshold(col, "p_") for col in prec_cols])
        prec_values = row[prec_cols].values.astype(float)
        #print("P thresholds: ", prec_thresholds)
        
        rec_thresholds = np.array([extract_threshold(col, "r_") for col in rec_cols])
        rec_values = row[rec_cols].values.astype(float)
        #print("R thresholds: ", rec_thresholds)
        
        # Store the curves
        mcc_curves.append((mcc_thresholds, mcc_values))
        precision_curves.append((prec_thresholds, prec_values))
        recall_curves.append((rec_thresholds, rec_values))

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
    # Main plotting area
    ax_main = fig.add_subplot(gs[0, 0])
    # "Empty" subplot for the legend
    ax_legend = fig.add_subplot(gs[0, 1])
    # Plot all MCC curves in the main subplot
    for (thresh, values), label in zip(mcc_curves, file_labels):
        ax_main.plot(thresh, values, label=label)
    # Configure the main subplot
    ax_main.set_xlabel("Threshold")
    ax_main.set_ylabel("MCC")
    ax_main.set_ylim(0, 0.6)
    ax_main.set_title("MCC vs. Threshold")
    ax_main.grid(True)
    # Move the legend into the second subplot
    handles, labels = ax_main.get_legend_handles_labels()
    ax_legend.axis('off')  # hide axis lines/ticks
    ax_legend.legend(handles, labels, loc='upper left', ncol=3)
    # Save before showing (recommended)
    plt.savefig("anaysis/plots/plot_mcc_after.png", dpi=300, bbox_inches="tight")
    # Finally, show the figure
    plt.show()

    # Plot Precision for all files in one plot
    plt.figure(figsize=(8, 6))
    for (thresh, values), label in zip(precision_curves, file_labels):
        plt.plot(thresh, values, label=label)
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title("Precision vs. Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("anaysis/plots/plot_p.png", dpi=300, bbox_inches="tight")

    # Plot Recall for all files in one plot
    plt.figure(figsize=(8, 6))
    for (thresh, values), label in zip(recall_curves, file_labels):
        plt.plot(thresh, values, label=label)
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title("Recall vs. Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("anaysis/plots/plot_r.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # List of file paths and corresponding labels for each file.
    file_paths = DoTA_dirs[12:] # replace with your actual file paths
    file_paths = [DoTA_dirs[i] for i in (28, 32, 36, 40, 44, 48)]
    file_labels = [os.path.basename(os.path.dirname(fp)).split("_")[0] for fp in file_paths]  # labels for the curves
    print(file_labels)

    plot_mcc_p_r_curves(file_paths=file_paths, file_labels=file_labels)


