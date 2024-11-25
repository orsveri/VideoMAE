import os
import zipfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted


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

    def init_plot_data(self, preds, labels):
        timeframes = list(range(len(preds)))

        preds = [p[-1] for p in preds]
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
            subplots_data[0]["data_items"].append({"x": timeframes, "y": preds, "label": "probs",
                                              "color": "blue_p", "labels_color": "blue_p", "threshold_areas": True})
        if labels is not None:
            subplots_data[0]["data_items"].append({"x": timeframes, "y": preds, "label": "gt labels",
                                                   "color": "black", "labels_color": "black", "threshold_areas": True})
            unsafe = np.array(labels).astype(bool)
            for sd in subplots_data:
                sd["fill_between"] = {"x": timeframes, "where": unsafe, "color": colors["red_l"]}

        self.subplots_data = subplots_data

    def get_frame(self, image_clip, cur_image, curr_frame=None, curr_probs=None, curr_label=None):
        im_h, im_w, _ = image_clip.shape
        fh_vid = self.fw * im_h / im_w
        fh_cur = 0.5 * self.fw * im_h / im_w
        nrows = len(self.subplots_data)
        fh = self.fh_base * nrows + self.fh_add + fh_vid
        scale = (im_w + self.border_pixels) / self.fw

        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100

        px = 1 / plt.rcParams['figure.dpi']
        figsize = (scale * self.fw * px, scale * fh * px)
        height_ratios = [fh_vid * 1.1, fh_cur * 1.1]
        height_ratios.extend([self.fh_base for _ in range(nrows)])

        fig, axes = plt.subplots(nrows + 2, 1, figsize=figsize, height_ratios=height_ratios, sharex=False,
                                 constrained_layout=True)

        extra = ""
        fig.suptitle(
            self.video_title + f"\n{extra}T: {curr_frame}, P: {round(curr_probs, 2)}, L: {curr_label}",
            fontsize=self.highlighted_fsize
        )
        fig.subplots_adjust(top=0.8)

        vid_ax = axes[0]
        cur_frame_ax = axes[1]
        axes = axes[2:]

        # Put image in the plot (imshow breaks resolution)
        vid_ax.pcolor(Image.fromarray(cv2.cvtColor(image_clip[::-1, :, :], cv2.COLOR_BGR2RGB)))
        vid_ax.axis('off')
        cur_frame_ax.pcolor(Image.fromarray(cv2.cvtColor(cur_image[::-1, :, :], cv2.COLOR_BGR2RGB)))
        cur_frame_ax.axis('off')

        for ax, sp_data in zip(axes, self.subplots_data):
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
            with warnings.catch_warnings(action="ignore"):
                ax.legend(loc="lower left")
        for i in range(1, len(axes)):
            axes[i].sharex(axes[0])

        axes[0].set_xticklabels([])

        with warnings.catch_warnings(action="ignore"):
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
            if save:
                os.makedirs(out_name.parent, exist_ok=True)

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

            if save:
                cv2.imwrite(str(out_name), output_img)
            else:
                batch_vis.append(output_img)
        if save:
            return None
        else:
            return batch_vis


def get_clip_data(clip_name, df):
    clip_data = df[df['clip'] == clip_name].copy()
    return clip_data


def save_video_dota(clip_dir, probs, labels, filenames, seq_length, out_path):
    # 1. Find unused labels in the beginning of the clip
    frames = []
    zip_file_path = os.path.join(clip_dir, "images.zip")
    with zipfile.ZipFile(zip_file_path, 'r') as archive:
        all_filenames = natsorted([img for img in archive.namelist() if img.endswith(".jpg")])
        intersection = set(all_filenames).intersection(set(filenames))
        assert natsorted(list(intersection)) == filenames
        # 2. Prepare plot template
        vis = VisFrameConstructor()
        vis.init_plot_data(preds=probs, labels=labels)
        # 2. Read all images and format them
        for fn in all_filenames:
            with archive.open(fn) as img_file:
                img_data = img_file.read()
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
                frames.append(img)
    # 3. Open the video stream, save frames and close the stream
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'MP4V'),
        10.,
        (frames[0].shape[1], frames[0].shape[0])
    )
    for frame in frames:
        img = vis.get_frame(
            image=frame,
            curr_frame=current_timestep,
            curr_probs=current_probs,
            curr_label=current_label
        )
        writer.write(img)
    writer.release()


predictions = "/home/sorlova/repos/NewStart/VideoMAE/logs/dota_fixloss/focal_1gpu/OUT_DoTA/predictions_0.csv"
out_folder = "/home/sorlova/repos/NewStart/VideoMAE/logs/dota_fixloss/focal_1gpu/OUT_DoTA_newvids/"
video_dir = "/mnt/experiments/sorlova/datasets/DoTA/frames"
seq_length = 16

df = pd.read_csv(predictions)
clips = natsorted(pd.unique(df['clip']))
random_clip_name = clips[0]
clip_data = get_clip_data(random_clip_name, df)

logits = clip_data[["logits_safe", "logits_risk"]].values  # Shape (N, 2)
logits_tensor = torch.tensor(logits, dtype=torch.float32)
probabilities = torch.nn.functional.softmax(logits_tensor, dim=1).numpy()

save_video_dota(
    clip_dir=os.path.join(video_dir, random_clip_name),
    probs=probabilities[:, 1],
    labels=clip_data["label"].values,
    filenames=clip_data["filename"].tolist(),
    seq_length=seq_length,
    out_path=os.path.join(out_folder, random_clip_name)
)

