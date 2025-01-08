import os
import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import natsort as ns


def get_anomaly_ranges(labels):
    """
    Calculates the start and stop frames for contiguous anomaly ranges in video labels.

    Args:
        labels (list or np.ndarray): Binary labels for video frames (0 for normal, 1 for anomaly).

    Returns:
        list of tuples: Each tuple contains (start_frame, stop_frame) for an anomaly range.
    """
    # Ensure the labels are a numpy array for efficient processing
    labels = np.array(labels)
    # Find indices where the label changes
    change_points = np.diff(labels, prepend=0, append=0)  # Extend to detect range at edges
    # Find the start (1) and stop (-1) points
    start_frames = np.where(change_points == 1)[0]
    stop_frames = np.where(change_points == -1)[0] - 1
    # Return the ranges as (start_frame, stop_frame)
    return list(zip(start_frames, stop_frames))


def generate_labels(length, anomaly_ranges):
    """
    Generates a binary list of labels with given anomaly ranges.

    Args:
        length (int): The total length of the binary label list.
        anomaly_ranges (list of tuples): List of (start_frame, stop_frame) tuples where labels should be 1.

    Returns:
        list: Binary label list with anomalies set to 1.
    """
    # Initialize the label list with 0s
    labels = [0] * length
    # Set labels to 1 for each anomaly range
    for start, stop in anomaly_ranges:
        labels[start:stop + 1] = [1] * (stop - start + 1)
    return labels


def refine_labeling(anno_path, set_old_ranges, new_ranges):
    with open(anno_path, "r") as f:
        anno = json.load(f)
    anno_labels = deepcopy(anno["labels"])
    clip_label_id = anno["accident_id"]
    clip_label_name = anno["accident_name"]
    bin_labels = [1 if int(frame_label["accident_id"]) > 0 else 0 for frame_label in anno["labels"]]

    # calculate old range(s)
    set_old_ranges = [(int(10 * r[0]), int(10 * r[1])) for r in set_old_ranges]
    new_ranges = [(int(10 * r[0]), int(10 * r[1])) for r in new_ranges]
    old_ranges = get_anomaly_ranges(bin_labels)
    assert old_ranges == set_old_ranges

    anno["anomaly_start"] = new_ranges[0][0]
    anno["anomaly_end"] = new_ranges[0][1]

    # make new range(s)
    new_bin_labels = generate_labels(length=len(bin_labels), anomaly_ranges=new_ranges)
    assert len(anno_labels) == len(bin_labels)

    for al, bl in zip(anno_labels, new_bin_labels):
        if bl == 0:
            al["accident_id"] = 0
            al["accident_name"] = "normal"
        elif bl == 1:
            al["accident_id"] = clip_label_id
            al["accident_name"] = clip_label_name
        else:
            raise ValueError(f"Invalid label. Can be only 0 or 1 but given: {bl}. Info:\n{al}")

    anno["labels"] = anno_labels
    # save the updated annotation
    with open(anno_path, "w") as f:
        json.dump(anno, f, indent=2)


def prepare_csv(anno_dir, clips, save_to):
    clip_ranges = []
    clips = ns.natsorted(clips, alg=ns.ns.IGNORECASE)
    for clip in clips:
        anno_path = os.path.join(anno_dir, f"{clip}.json")
        with open(anno_path, "r") as f:
            anno = json.load(f)
        bin_labels = [1 if int(frame_label["accident_id"]) > 0 else 0 for frame_label in anno["labels"]]
        old_ranges = get_anomaly_ranges(bin_labels)
        clip_ranges.append(old_ranges)
    N = max([len(r) for r in clip_ranges])
    assert N == 1
    starts = [r[0][0]/10 for r in clip_ranges]
    ends = [r[0][1]/10 for r in clip_ranges]
    df = pd.DataFrame({"clip": clips, "old_start": starts, "old_end": ends})
    df["new_start"] = None
    df["new_end"] = None
    df.to_csv(save_to, header=True, index=True)


if False:
    # manual change
    anno_dir = "/mnt/experiments/sorlova/datasets/DoTA_refined/dataset/annotations"
    clip_name = "bFGmOp9H3MA_000355_p2"
    set_old_ranges = [(0.0,11.1)]  # index number of labeled frames starting from 0
    new_ranges = [(3.7,7.2)]  # index number of labeled frames starting from 0
    # read anno
    anno_path = os.path.join(anno_dir, f"{clip_name}.json")
    refine_labeling(anno_path=anno_path, set_old_ranges=set_old_ranges, new_ranges=new_ranges)

if False:
    # prepare csv for relabeling
    clips = [os.path.splitext(item)[0] for item in os.listdir("/mnt/experiments/sorlova/AITHENA/NewStage/VideoMAE_results/auroc_behaviour_vis/crossentropy/cleaning/allval_ckpt-1_bad03")]
    prepare_csv(anno_dir, clips, save_to="/mnt/experiments/sorlova/datasets/DoTA_refined/dataset/example.csv")

if False:
    anno_dir = "/mnt/experiments/sorlova/datasets/DoTA_refined/dataset/annotations"
    fixes_csv = "/mnt/experiments/sorlova/datasets/DoTA_refined/dataset/fix_train1.csv"
    fixes = pd.read_csv(fixes_csv)
    for i, row in tqdm(fixes.iterrows(), total=len(fixes), desc="Processing clips"):
        clip_name = row["clip"]
        set_old_ranges = [(row["old_start"], row["old_end"])]  # index number of labeled frames starting from 0
        new_ranges = [(row["new_start"], row["new_end"])]  # index number of labeled frames starting from 0
        # read anno
        anno_path = os.path.join(anno_dir, f"{clip_name}.json")
        refine_labeling(anno_path=anno_path, set_old_ranges=set_old_ranges, new_ranges=new_ranges)