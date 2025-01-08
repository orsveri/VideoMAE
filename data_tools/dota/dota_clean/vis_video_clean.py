import os
import json
import zipfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec
from deepspeed.ops.sparse_attention.trsrc import fname
from natsort import natsorted
from tqdm import tqdm


colors = {
    'green_p': '#006600',
    'green_l': '#bddbbd',
    'purple_p': '#8F4289',
    'purple_l': '#dfb9dc',
    'blue_p': '#134475',
    'blue_l': '#a7ccf1',
    'red_p': '#751320',
    'red_l': '#deb8bd',
    'curr_frame': '#71797e',
    'brown_grid': '#5c2b0a',
    'black': 'k',
}

FPS = 10


class VisFrameConstructor():
    gap_color = [0, 0, 180]
    bbox_color = (0, 180, 0)
    normal_fsize = 12
    highlighted_fsize = 14
    transparency = 0.3
    fw = 20
    border_pixels = 20
    fh_base = 3
    fh_add = 1
    c_safe = (0, 120, 0)
    c_unsafe = (20, 0, 180)
    lw = 1.5
    curr_lw = 3
    risk_threshold = 0.5

    def init_plot_data(self, preds, labels, empty_start_preds=0, empty_start_labels=0):
        timeframes_preds = [(item+empty_start_preds)/FPS for item in list(range(len(preds)))]
        timeframes_labels = [(item + empty_start_labels) / FPS for item in list(range(len(labels)))]

        #preds = [p[-1] for p in preds]
        subplots_data = [
            {
                "subplot_name": "Risk",
                "x_label": "timesteps",
                "y_label": "score",
                "data_items": [],
            },
            # {
            #     "subplot_name": "Velocities and times",
            #     "x_label": "timesteps",
            #     "y_label": "s",
            #     "data_items": [],
            # },
        ]

        if preds is not None:
            subplots_data[0]["data_items"].append({"x": timeframes_preds, "y": preds, "label": "probs",
                                              "color": colors["blue_p"], "labels_color": colors["red_l"], "threshold_areas": True})
        if labels is not None:
            subplots_data[0]["data_items"].append({"x": timeframes_labels, "y": labels, "label": "gt labels",
                                                   "color": colors["black"], "labels_color": colors["black"], "threshold_areas": True})
            unsafe = np.array(labels).astype(bool)
            for sd in subplots_data:
                sd["fill_between"] = {"x": timeframes_labels, "where": unsafe, "color": colors["red_l"]}

        self.subplots_data = subplots_data
        self.video_title = ""

    def get_frame(self, image_clip, cur_image, curr_frame=None, curr_probs=None, curr_label=None):
        im_h, im_w, _ = image_clip.shape
        fh_vid = self.fw * im_h / im_w
        fh_cur = 2 * self.fw * im_h / im_w
        nrows = len(self.subplots_data)
        fh = self.fh_base * nrows + self.fh_add + fh_vid + fh_cur
        scale = (im_w + self.border_pixels) / self.fw

        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100

        px = 1 / plt.rcParams['figure.dpi']
        figsize = (scale * self.fw * px, scale * fh * px)
        height_ratios = [fh_vid * 1.1, fh_cur * 1.1]
        height_ratios.extend([self.fh_base for _ in range(nrows)])

        fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = fig.add_gridspec(nrows + 2, 3, height_ratios=height_ratios, width_ratios=[0.25, 0.5, 0.25])

        # fig, axes = plt.subplots(nrows + 2, 1, figsize=figsize, height_ratios=height_ratios, sharex=False,
        #                          constrained_layout=True)
        axes = []
        vid_ax = fig.add_subplot(gs[0, :])
        middle_left = fig.add_subplot(gs[1, 0])
        middle_left.axis('off')
        cur_frame_ax = fig.add_subplot(gs[1, 1])
        middle_right = fig.add_subplot(gs[1, 2])
        middle_right.axis('off')
        for i in range(nrows):
            axes.append(fig.add_subplot(gs[2+i, :]))

        extra = ""
        fig.suptitle(
            self.video_title + f"\n{extra}T: {str(round(curr_frame, 1))}, P: {str(round(curr_probs, 3))}, L: {curr_label}",
            fontsize=self.highlighted_fsize
        )
        #fig.subplots_adjust(top=0.8)

        # Put image in the plot (imshow breaks resolution)
        vid_ax.pcolor(Image.fromarray(cv2.cvtColor(image_clip[::-1, :, :], cv2.COLOR_BGR2RGB)))
        vid_ax.axis('off')
        cur_frame_ax.pcolor(Image.fromarray(cv2.cvtColor(cur_image[::-1, :, :], cv2.COLOR_BGR2RGB)))
        cur_frame_ax.axis('off')

        for ax, sp_data in zip(axes, self.subplots_data):
            ax.set_xlim(0, sp_data["data_items"][0]["x"][-1])
            ax.set_ylim(-0.1, 1.1)
            for data_item in sp_data["data_items"]:
                ax.plot(data_item["x"], data_item["y"], label=data_item["label"], color=data_item["color"],
                        linewidth=self.lw)
            if "twinx_data" in sp_data:
                twinx_data = sp_data["twinx_data"]
                ax2 = ax.twinx()
                ax2.set_ylabel(twinx_data["label"], color=twinx_data["labels_color"], fontsize=self.normal_fsize,
                               weight='normal')
                ax2.plot(twinx_data["x"], twinx_data["y"], color=twinx_data["color"], linewidth=self.lw)
                ax2.tick_params(axis='y', labelcolor=twinx_data["labels_color"])
                ax2.grid(color=colors["brown_grid"])
                # threshold
                if twinx_data.get("threshold_areas"):
                    xs = ax2.get_xlim()
                    ax2.fill_between(x=xs, y1=0, y2=self.risk_threshold, color='green', interpolate=True, alpha=.15)
                    ax2.fill_between(x=xs, y1=self.risk_threshold, y2=1., color='red', interpolate=True, alpha=.15)
            ymin, ymax = ax.get_ylim()
            if "fill_between" in sp_data:
                fill_data = sp_data["fill_between"]
                ax.fill_between(
                    fill_data["x"], ymin, ymax, where=fill_data["where"], color=fill_data["color"],
                    alpha=self.transparency
                )
            if curr_frame is not None:
                ax.vlines(
                    x=curr_frame, ymin=ymin, ymax=ymax,
                    colors=colors["curr_frame"], linewidth=self.curr_lw, linestyles="dotted"
                )
            ax.set_title(sp_data["subplot_name"])
            ax.set_xlabel(sp_data["x_label"], fontsize=self.normal_fsize, weight='normal')
            ax.set_ylabel(sp_data["y_label"], fontsize=self.normal_fsize, weight='normal')
            ax.grid()
            with warnings.catch_warnings():
                ax.legend(loc="lower left")

        for i in range(1, len(axes)):
            axes[i].sharex(axes[0])

        #axes[0].set_xticklabels([])
        loc = plticker.MultipleLocator(base=0.5)  # this locator puts ticks at regular intervals
        axes[0].xaxis.set_major_locator(loc)

        with warnings.catch_warnings():
            plt.legend()
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        [ax.cla() for ax in axes]
        plt.close(fig)
        return img

    def create_sequence_vis(self, batch_inputs, batch_seq_fnames):
        batch_size = batch_inputs[0].shape[0]
        seq_len = batch_inputs[0].shape[-3]  # (B, C, T, H, W)
        batch_vis = []
        for bi in range(batch_size):
            data_id = batch_seq_fnames["data_id"][bi]
            last_filename = batch_seq_fnames["filenames"][-1][bi]  # ! nested list shape is (seq_len, batch) !
            out_name = self.out_vis_dir / f"{self.tag}{data_id}" / f"{Path(last_filename).stem}.jpg"

            output_img = []
            img_mod = batch_inputs[self.camera_idx][bi]
            gap_v = np.ones(shape=(img_mod.shape[-2], 5, 3)) * np.array([self.gap_color])
            row = []
            for i in range(seq_len):
                img = np.rint(torch.moveaxis(img_mod[:, i, :, :], 0, -1).cpu().numpy() * 255).astype(np.uint8)
                img = np.ascontiguousarray(img, dtype=np.uint8)
                if self.bbox_idx is not None:
                    # add bboxes
                    batch_bboxes, batch_classes, batch_actor_ids = batch_inputs[self.bbox_idx][i]
                    bboxes = batch_bboxes[bi].cpu().tolist()
                    classes = batch_classes[bi].cpu().tolist()
                    if len(bboxes) > 0:
                        img = self.add_bbox_to_rgb(img, bboxes=bboxes, classes=classes)
                if i < seq_len - 1:
                    img = np.hstack((img, gap_v))
                row.append(img)
            row = np.hstack(row)
            output_img.append(row)

            output_img = np.rint(np.vstack(output_img)).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            batch_vis.append(output_img)

        return batch_vis


def get_clip_data(clip_name, df):
    clip_data = df[df['clip'] == clip_name].copy()
    return clip_data


def get_image_clip(images, cur_idx, seq_length, fps=FPS):
    t = 4
    step = round(seq_length / (t-1))
    clip_ids = [round(cur_idx - step*i) + 1 for i in range(t-1, -1, -1)]
    clip_ids[0] = cur_idx - seq_length + 1
    clip_ids[-1] = cur_idx
    image_clip = []
    for cid in clip_ids:
        if cid < 0:
            img = np.zeros_like(images[cur_idx])
        else:
            img = images[cid]
        image_clip.append(img)
    image_clip = cv2.resize(np.hstack(image_clip), (0, 0), fx=0.5, fy=0.5)
    return image_clip


def save_video_dota(clip_dir, probs, labels, filenames, seq_length, out_path, img_ext = ".jpg"):
    #
    if os.path.exists(out_path):
        return
    # 1. Find unused labels in the beginning of the clip
    anno_path = os.path.join(os.path.dirname(os.path.dirname(clip_dir)), "dataset", "annotations", f"{os.path.basename(clip_dir)}.json")
    with open(anno_path) as f:
        anno = json.load(f)
        # sort is not required since we read already sorted timesteps from annotations
        all_timesteps = natsorted([int(os.path.splitext(os.path.basename(frame_label["image_path"]))[0]) for frame_label
                               in anno["labels"]])
        all_labels = [1 if int(frame_label["accident_id"]) > 0 else 0 for frame_label in anno["labels"]]
    # 2. Read frames to form plots
    frames = []
    all_filenames = []
    zip_file_path = os.path.join(clip_dir, "images.zip")
    with zipfile.ZipFile(zip_file_path, 'r') as archive:
        all_filenames = natsorted([img for img in archive.namelist() if img.endswith(img_ext)])
        intersection = set(all_filenames).intersection(set(filenames))
        assert natsorted(list(intersection)) == filenames, f"len filenames: {len(intersection)}, len intersection {len(filenames)}"
        last_labeled_id = all_filenames.index(f"{str(all_timesteps[-1]).zfill(6)}.jpg")
        all_filenames = all_filenames[:last_labeled_id+1]
        # 2. Prepare plot template
        vis = VisFrameConstructor()
        vis.init_plot_data(preds=probs, labels=all_labels, empty_start_preds=15, empty_start_labels=0)
        # 2. Read all images and format them
        for fn in all_filenames:
            with archive.open(fn) as img_file:
                img_data = img_file.read()
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
                frames.append(img)
    # 3. Open the video stream, save frames and close the stream
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = None
    for i, (fname, frame) in enumerate(tqdm(zip(all_filenames, frames), total=len(all_filenames), leave=False)):
        if fname in filenames:
            i_ = filenames.index(fname)
        else:
            i_ = None
        clip_imgs = get_image_clip(images=frames, cur_idx=i, seq_length=seq_length)
        img = vis.get_frame(
            image_clip=clip_imgs,
            cur_image=frame,
            curr_frame=i/FPS,
            curr_probs=-1 if i_ is None else probs[i_],
            curr_label=all_labels[all_timesteps.index(fname)] if fname in all_timesteps else -1
        )
        if writer is None:
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*'MP4V'),
                10.,
                (img.shape[1], img.shape[0])
            )
        writer.write(img)
        #cv2.imwrite("/mnt/experiments/sorlova/AITHENA/NewStage/VideoMAE_results/auroc_behaviour_vis/crossentropy/checkpoint-15_OUT/debug.jpg", img)
    writer.release()


ckpt = 15
tag = "allval"  # "_train
version = "crossentropy"
report_csv_clipnames = f"/home/sorlova/repos/NewStart/VideoMAE/logs/clean_datasets/DoTA/b32x2x1gpu_ce_VAL/checkpoint-{ckpt}/OUTval/err_report_bad0.025.csv"
predictions = f"/home/sorlova/repos/NewStart/VideoMAE/logs/clean_datasets/DoTA/b32x2x1gpu_ce_VAL/checkpoint-{ckpt}/OUTval/predictions_0.csv"
out_folder = f"/mnt/experiments/sorlova/AITHENA/NewStage/VideoMAE_results/auroc_behaviour_vis/crossentropy/cleaning/{tag}_ckpt-{ckpt}_bad0025"
#video_dir = "/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA2000/frames" # "/mnt/experiments/sorlova/datasets/DoTA/frames"
video_dir = "/mnt/experiments/sorlova/datasets/DoTA/frames"
seq_length = 16

# ============
clip_df = pd.read_csv(report_csv_clipnames)
clip_names = clip_df["clip"]


for clip_name in tqdm(clip_names):
    df = pd.read_csv(predictions)
    clip_data = get_clip_data(clip_name, df)

    logits = clip_data[["logits_safe", "logits_risk"]].values  # Shape (N, 2)
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probabilities = torch.nn.functional.softmax(logits_tensor, dim=1).numpy()

    save_video_dota(
        clip_dir=os.path.join(video_dir, clip_name),
        probs=probabilities[:, 1],
        labels=clip_data["label"].values,
        filenames=clip_data["filename"].tolist(),
        seq_length=seq_length,
        out_path=os.path.join(out_folder, clip_name + ".mp4"),
        img_ext=".jpg"
    )

