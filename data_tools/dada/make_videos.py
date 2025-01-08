import os
import cv2
import zipfile
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


data_dir_dict = {
    "dota": "/mnt/experiments/sorlova/datasets/DoTA/frames",
}


def find_incorrect_clips(df_data, out_dir, dataset="dota", save_histograms=True):
    clips = natsorted(pd.unique(df_data['clip']))
    ap_scores = []
    auroc_scores = []
    acc_scores = []
    best_f1 = []
    best_th = []

    logits = df_data[["logits_safe", "logits_risk"]].values  # Shape (N, 2)
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probabilities = torch.nn.functional.softmax(logits_tensor, dim=1).numpy()
    df_data['pred'] = probabilities[:, 1]

    for i, cl in enumerate(clips):
        clip_data = df_data[df_data['clip'] == cl].copy()
        labels = clip_data["label"].values
        pred  = clip_data["pred"].values
        predicted_labels = pred > 0.5

        has_both_labels = np.unique(labels).shape[0] > 1

        acc_score = (predicted_labels == labels).mean()
        acc_scores.append(acc_score)
        if has_both_labels:
            ap_score = average_precision_score(labels, pred)
            auroc_score = roc_auc_score(labels, pred)
            precision, recall, thresholds = precision_recall_curve(labels, pred)
            f1_scores = (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            ap_scores.append(ap_score)
            auroc_scores.append(auroc_score)
            best_f1.append(max(f1_scores))
            best_th.append(best_threshold)

    acc_scores = np.array(acc_scores)
    ap_scores = np.array(ap_scores)
    auroc_scores = np.array(auroc_scores)
    best_th = np.array(best_th)

    if save_histograms:
        # Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(acc_scores, bins=25, edgecolor='black')
        plt.xlabel('Per-clip ACC')
        plt.ylabel('Count')
        plt.title('Histogram of ACC values')
        plt.savefig(os.path.join(out_dir, f'err_{dataset}_histacc.png'), bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(8, 6))
        plt.hist(ap_scores, bins=25, edgecolor='black')
        plt.xlabel('Per-clip AP')
        plt.ylabel('Count')
        plt.title('Histogram of AP values')
        plt.savefig(os.path.join(out_dir, f'err_{dataset}_histap.png'), bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(8, 6))
        plt.hist(auroc_scores, bins=25, edgecolor='black')
        plt.xlabel('Per-clip AUROC')
        plt.ylabel('Count')
        plt.title('Histogram of AUROC values')
        plt.savefig(os.path.join(out_dir, f'err_{dataset}_histauroc.png'), bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(8, 6))
        plt.hist(best_th, bins=25, edgecolor='black')
        plt.xlabel('Per-clip best thresholds')
        plt.ylabel('Count')
        plt.title('Histogram of threshold values')
        plt.savefig(os.path.join(out_dir, f'err_{dataset}_histth.png'), bbox_inches='tight')
        plt.close()

    # Save clips with errors
    err_clips = [cl for i, cl in enumerate(clips) if acc_scores[i] < 0.4]
    err_acc = [acc_scores[i] for i, cl in enumerate(clips) if acc_scores[i] < 0.4]

    df = pd.DataFrame({"clip": err_clips, "clip_acc": err_acc})
    df = df.sort_values(by='clip_acc')
    df.to_csv(os.path.join(out_dir, f"errclips_val_{dataset}.csv"))
    print("Err clips: ", len(err_clips))

    video_out_dir = os.path.join(out_dir, f'err_videos_{dataset}')
    os.makedirs(video_out_dir, exist_ok=True)
    for i, row in tqdm(df.iterrows()):
        clip_data = df_data[df_data['clip'] == row['clip']]
        labels = clip_data['label'].to_list()
        preds = clip_data['pred'].to_list()
        filenames = clip_data['filename']
        save_video_zip(
            clip_name=row['clip'],
            score=row['clip_acc'],
            out_dir=video_out_dir,
            labels=labels,
            preds=preds,
            filenames=filenames,
            dataset=dataset
        )
    return df


def save_video_zip(clip_name, score, out_dir, labels, preds, filenames=None, dataset="dota"):
    data_dir = data_dir_dict[dataset]
    zip_file_path = os.path.join(data_dir, clip_name, "images.zip")
    out_name = os.path.join(out_dir, f"{clip_name}_acc{score}.mp4")
    with zipfile.ZipFile(zip_file_path, 'r') as archive:
        images = natsorted([img for img in archive.namelist() if img.endswith(".jpg")])
        with archive.open(images[0]) as img_file:
            img_data = img_file.read()
            first_image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            first_image = cv2.resize(first_image, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
            height, width, layers = first_image.shape

        if filenames is not None:
            images = natsorted(filenames)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        video = cv2.VideoWriter(out_name, fourcc, 10, (width, height))
        for i, image in enumerate(images):
            with archive.open(image) as img_file:
                img_data = img_file.read()
                frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            if i < len(labels):
                cv2.putText(frame, str(labels[i]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, str(labels[i]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 200, 0), 2, cv2.LINE_AA)
                rtext = str(round(preds[i], 2))
                text_size = cv2.getTextSize(rtext, cv2.FONT_HERSHEY_SIMPLEX, 1, 4)[0]
                cv2.putText(frame, rtext, (width - text_size[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, rtext, (width - text_size[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (200, 0, 200), 2, cv2.LINE_AA)
            video.write(frame)
        video.release()
        print(f"Video saved as {out_name}")


# predictions = "logs/dota_fixloss/focal_1gpu/OUT_DoTA/predictions_0.csv"
# out_folder = "logs/dota_fixloss/focal_1gpu/OUT_DoTA/"
#
# df = pd.read_csv(predictions)
#
# find_incorrect_clips(df_data=df, out_dir=out_folder, save_histograms=False)




