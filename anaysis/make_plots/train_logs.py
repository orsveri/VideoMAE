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


color_dict = {
    "VidNeXt": "blue",
    "ResNet+NST": "magenta",
    "ConvNeXt": "black",
    "R(2+1)D": "blue",
    "MOVAD": "purple",
    "VideoMAE-S": "green",
    "VideoMAE2-S": "orange",
    "InternVideo2-S": "orange",
    "ViViT-B": "red",
    # add as many as you need
}

marker_dict = {
    "VidNeXt": "D",     # diamond
    "ResNet+NST": "v",   # triangle down
    "ConvNeXt": "v",    # triangle down
    "R(2+1)D": "s",       # square
    "MOVAD": "^",       # triangle up
    "VideoMAE-S": "*",         # star
    "VideoMAE2-S": "^",         # triangle up
    "InternVideo2-S": "D",         # diamond
    "ViViT-B": "o",         # circle
}

markersize_dict = {
    "VidNeXt": 30,      #
    "ResNetNST": 40,   #
    "ConvNeXt": 40,    #
    "R(2+1)D": 30,
    "MOVAD": 40,       #
    "VideoMAE-S": 100,         #
    "VideoMAE2-S": 46,         #
    "InternVideo2-S": 30,         #
    "ViViT-B": 35,         #
}

linestyle_dict = {
    "VidNeXt": (0, (5, 1)),     
    "ResNet+NST": (0, (5, 1)),   
    "ConvNeXt": (0, (3, 1, 1, 1)),    
    "R(2+1)D": (0, (5, 1)),      
    "MOVAD": "solid",       
    "VideoMAE-S": "solid",         
    "VideoMAE2-S": "solid",         
    "InternVideo2-S": "solid",         
    "ViViT-B": "solid",        
}
linestyle_marker_dict = {
    "VidNeXt": "D",     # diamond
    "ResNet+NST": None,   # triangle down
    "ConvNeXt": None,    # triangle down
    "R(2+1)D": None,       # square
    "MOVAD": None,       # triangle up
    "VideoMAE-S": None,         # star
    "VideoMAE2-S": None,         # triangle up
    "InternVideo2-S": "D",         # diamond
    "ViViT-B": None,         # circle
}