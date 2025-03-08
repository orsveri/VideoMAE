import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from matplotlib import pyplot as plt

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from anaysis.metrics import calculate_metrics, calculate_fn_group, calculate_fn_group_thresholds, calculate_MORE_metrics, THRESHOLDS


def save_metrics(anno_csv, preds_dir):
    pred_csv = os.path.join(preds_dir, "predictions.csv")
    out_results_group = os.path.join(preds_dir, "group_stats.csv")
    out_metrics_thresholded = os.path.join(preds_dir, "thresh_stats.csv")

    # 1 combine data
    df_preds = pd.read_csv(pred_csv)
    df_anno = pd.read_csv(anno_csv)

    # Prepare probs
    logits = df_preds[["logits_safe", "logits_risk"]].to_numpy()
    probs = np.exp(logits)/np.expand_dims(np.sum(np.exp(logits), axis=-1), axis=-1)
    probs = probs[:, 1]
    df_preds["probs"] = probs
    df_preds = df_preds.drop('logits_safe', axis=1)
    df_preds = df_preds.drop('logits_risk', axis=1)

    if "dota" in os.path.basename(preds_dir).lower():
        df_preds['filename'] = df_preds['filename'].apply(lambda x: f"{str(x).zfill(6)}.jpg" if isinstance(x, int) else x)
        ok_percent = 0.3
    elif "dada2k" in os.path.basename(preds_dir).lower():
        df_preds['filename'] = df_preds['filename'].apply(lambda x: f"{str(x).zfill(4)}.png" if isinstance(x, int) else x)
        if "movad" in preds_dir.lower():  # because 30  -> 10 FPS and frame-by-frame prediction
            ok_percent = 0.3
        elif "comparison_cc" in preds_dir.lower():  # because 30  -> 10 FPS
            ok_percent = 0.15
        else:
            ok_percent = 0.15
    else:
        raise ValueError(f"Unknown dataset in {os.path.basename(preds_dir).lower()}")

    annot_subset = df_anno[['clip', 'filename', 'ego', 'night', 'cat']]
    df_preds = df_preds.merge(annot_subset, on=['clip', 'filename'], how='left')

    if df_preds.isnull().values.any():
        num_rows = df_preds.shape[0]
        num_rows_with_nan = df_preds.isnull().any(axis=1).sum()
        percent = num_rows_with_nan / num_rows
        print(f"There are {percent*100:.2f}% missing values in the DataFrame while normal max percent is {ok_percent}. {preds_dir}")
        if percent < ok_percent:
            print("It's okay. Remove invalid rows and proceed...")
            df_preds = df_preds.dropna()
        else:
            print("It's not okay. Halt.")
            exit(0)
    else:
        print("No missing values found!")

    del annot_subset
    del df_anno

    # Data is ready
    results = []
    thresh_results = []
    empty_list = [-1 for _ in THRESHOLDS]

    # First, get general (non-grouped) metrics
    thresh_metrics = calculate_MORE_metrics(preds=df_preds["probs"], labels=df_preds["label"])
    mcc_thresholded_vals, p_thresholded_vals, r_thresholded_vals, acc_thresholded_vals, f1_thresholded_vals = thresh_metrics[-5:]
    thresh_results.append(
        ["all_samples"] + mcc_thresholded_vals + p_thresholded_vals + r_thresholded_vals + 
        acc_thresholded_vals + f1_thresholded_vals
        )

    # 2 get metrics
    df_preds["cat"] = df_preds["cat"].astype(int)
    df_preds["ego"] = df_preds["ego"].astype(int)
    df_preds["night"] = df_preds["night"].astype(int)

    categories = natsorted(df_preds["cat"].unique())
    for cat in categories:
        df_group = df_preds[df_preds["cat"] == cat]
        res = calculate_fn_group_thresholds(probs=df_group["probs"], labels=df_group["label"])
        res = np.array(res)
        res = res if cat == 0 else 1.-res
        thresh_results.append([f"cat_{cat}"] + empty_list + empty_list + list(res) + empty_list + empty_list)

    df_group = df_preds[df_preds['ego'] == 1]
    res = calculate_fn_group_thresholds(probs=df_group["probs"], labels=df_group["label"])
    res = 1. - np.array(res)
    thresh_results.append(["ego"] + empty_list + empty_list + list(res) + empty_list + empty_list)

    df_group = df_preds[df_preds['ego'] == 0]
    res = calculate_fn_group_thresholds(probs=df_group["probs"], labels=df_group["label"])
    res = 1. - np.array(res)
    thresh_results.append(["no_ego"] + empty_list + empty_list + list(res) + empty_list + empty_list)

    df_group = df_preds[df_preds['night'] == 1]
    res_ = calculate_MORE_metrics(preds=df_group["probs"], labels=df_group["label"])
    res = res_[:6]
    thresh_res = res_[-5:]
    results.append(["night"] + list(res))
    thresh_results.append(["night"] + thresh_res[0] + thresh_res[1] + thresh_res[2] + thresh_res[3] + thresh_res[4])

    df_group = df_preds[df_preds['night'] == 0]
    res_ = calculate_MORE_metrics(preds=df_group["probs"], labels=df_group["label"])
    res = res_[:6]
    thresh_res = res_[-5:]
    results.append(["day"] + list(res))
    thresh_results.append(["day"] + thresh_res[0] + thresh_res[1] + thresh_res[2] + thresh_res[3] + thresh_res[4])

    df = pd.DataFrame(results, columns=['group', 'acc', 'p', 'r', 'f1', 'map', 'auc'])
    df_thresh = pd.DataFrame(thresh_results, columns=["group"] + [f"mcc_{t:.2f}" for t in THRESHOLDS] + [f"p_{t:.2f}" for t in THRESHOLDS] + [f"r_{t:.2f}" for t in THRESHOLDS] + [f"acc_{t:.2f}" for t in THRESHOLDS] + [f"f1_{t:.2f}" for t in THRESHOLDS])
    df.to_csv(out_results_group)
    df_thresh.to_csv(out_metrics_thresholded)

    print(f"Done! Saved to: {out_results_group} and {out_metrics_thresholded}")


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
    # Direct finetuning of ViFM
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/1_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_16",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/2_dotaH_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/3_dada_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl1/4_dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_9",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/5_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_5",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/6_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_8",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/7_dada_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_4",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl2/8_dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_5",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/9_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_5",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/10_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_6",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/11_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_3",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl3/12_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_1",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/13_VITB_dota_lr5e4_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_9",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/14_VITB_dotah_lr5e4_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/15_VITB_dada_lr5e4_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_6",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/finetune/baselines/bl4/16_VITB_dadah_lr5e_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_bestap",
    # "Ablations"
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C1_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_25",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C2_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C3_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_bestap",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1k700/C4_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_bestap",
    # 
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/101_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/102_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/103_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_8",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd/104_dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_8",
    # Finetuning after SSPPT
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/201_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_11",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/202_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_13",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/203_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_8",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl1/pt_bdd-capdata/204_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_13",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/205_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_16",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/206_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/207_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_15",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl2/208_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_8",
    #
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/209_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_25",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/210_dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3/eval_DoTA_ckpt_14",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/211_dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_11",
    "/gpfs/work3/0/tese0625/VideoMAE_results/train_logs/ft_after_pretrain/bl3/212_dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3/eval_DoTA_ckpt_11",
    #
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


if __name__ == "__main__":
    DoTA_anno = "/gpfs/work3/0/tese0625/RiskNetData/DoTA_refined/dataset/frame_level_anno_val.csv"
    DADA2K_anno = "/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/DADA2K/DADA2K_my_split/frame_level_anno_val.csv"
    
    anno_csv = DoTA_anno
    preds_dirs = DoTA_dirs

    for pred_dir in tqdm(preds_dirs):
        save_metrics(anno_csv=anno_csv, preds_dir=pred_dir)


