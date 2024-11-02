import os
import cv2
import pandas as pd
from tqdm import tqdm
from natsort import natsorted


VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".flv")

HMDB51_categories = ('brush_hair', 'cartwheel', 'catch', 'chew', 'clap',
                     'climb', 'climb_stairs', 'dive', 'draw_sword', 'dribble',
                     'drink', 'eat', 'fall_floor', 'fencing', 'flic_flac',
                     'golf', 'handstand', 'hit', 'hug', 'jump',
                     'kick', 'kick_ball', 'kiss', 'laugh', 'pick',
                     'pour', 'pullup', 'punch', 'push', 'pushup',
                     'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball',
                     'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile',
                     'smoke', 'somersault', 'stand', 'swing_baseball', 'sword',
                     'sword_exercise', 'talk', 'throw', 'turn', 'walk',
                     'wave')



def get_filenames(hmdb_root):
    clips = natsorted([clip for clip in os.listdir(os.path.join(hmdb_root, "clips"))])
    correct_clips = []
    videos = []
    for clip in clips:
        clip_dir = os.path.join(hmdb_root, "clips", clip)
        if not os.path.isdir(clip_dir):
            continue
        clip_videos = natsorted([vid for vid in os.listdir(clip_dir) if os.path.splitext(vid)[1] in VID_EXT])
        videos.append(clip_videos)
        correct_clips.append(clip_dir)
    assert len(videos) == len(correct_clips)
    return correct_clips, videos


def get_filenames_from_splits(splits_folder, split_no):
    split_files = [f for f in os.listdir(splits_folder) if f"_split{split_no}.txt" in f]
    assert len(split_files) == 51
    categories = []
    train_videos = []
    test_videos = []
    for class_split in split_files:
        category = class_split.split('_test_split')[0]
        with open(os.path.join(splits_folder, class_split), 'r') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        train_vids = natsorted([line.split(" ")[0] for line in lines if line[-1]=="1"])
        test_vids = natsorted([line.split(" ")[0] for line in lines if line[-1] == "2"])
        assert len(train_vids) == 70
        assert len(test_vids) == 30
        train_videos.append(train_vids)
        test_videos.append(test_vids)
        categories.append(category)
    combined = sorted(zip(categories, train_videos, test_videos))
    categories, train_videos, test_videos = map(list, zip(*combined))
    return categories, train_videos, test_videos


def all_vids_exist(hmdb_root, categories, videos, video_folder="clips_avi"):
    for cat, vids in zip(categories, videos):
        for vid in vids:
            deb = os.path.join(hmdb_root, video_folder, cat, vid)
            if not os.path.exists(os.path.join(hmdb_root, video_folder, cat, vid)):
                return False
    return True


def all_vids_to_mp4(hmdb_root, categories, videos, inp_video_folder="clips_avi", out_video_folder="clips_mp4", desc=None):
    mp4_videos = []
    for cat, vids in tqdm(zip(categories, videos), total=len(categories), desc=desc):
        cat_mp4_videos = []
        for vid in tqdm(vids, desc=cat, leave=False):
            mp4_video = os.path.splitext(vid)[0] + ".mp4"
            inp_path = os.path.join(hmdb_root, inp_video_folder, cat, vid)
            out_path = os.path.join(hmdb_root, out_video_folder, cat, mp4_video)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            _to_mp4(inp_path=inp_path, out_path=out_path)
            cat_mp4_videos.append(mp4_video)
        mp4_videos.append(cat_mp4_videos)
    return mp4_videos


def _to_mp4(inp_path, out_path):
    cap = cv2.VideoCapture(inp_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_writer.write(frame)
    cap.release()
    video_writer.release()


def prepare_csv(hmdb_root, categories, videos, video_folder):
    rows = []
    for cat, vids in zip(categories, videos):
        for vid in vids:
            rows.append([os.path.join(hmdb_root, video_folder, cat, vid), HMDB51_categories.index(cat)])
    df = pd.DataFrame(rows, columns=["video_path", "label"])
    return df


# 0. Get files
dataset_root = "/mnt/experiments/sorlova/datasets/HMDB51"
splits_subfolder = "testTrainMulti_7030_splits"
split_no = 1

# 1. List train and test files
categories, train_videos, test_videos = get_filenames_from_splits(
    splits_folder=os.path.join(dataset_root, splits_subfolder),
    split_no=split_no
)
assert set(categories) == set(HMDB51_categories)
train_ok = all_vids_exist(hmdb_root=dataset_root, categories=categories, videos=train_videos, video_folder="clips_avi")
test_ok = all_vids_exist(hmdb_root=dataset_root, categories=categories, videos=test_videos, video_folder="clips_avi")
assert train_ok and test_ok
print("All filenames exist!")

# 1. Change the file format from .avi to .mp4
train_mp4_videos = all_vids_to_mp4(hmdb_root=dataset_root, categories=categories, videos=train_videos,
                            inp_video_folder="clips_avi", out_video_folder="clips_mp4", desc="TRAIN")
train_ok = all_vids_exist(hmdb_root=dataset_root, categories=categories, videos=train_mp4_videos, video_folder="clips_mp4")
test_mp4_videos = all_vids_to_mp4(hmdb_root=dataset_root, categories=categories, videos=test_videos,
                            inp_video_folder="clips_avi", out_video_folder="clips_mp4", desc="TEST")
test_ok = all_vids_exist(hmdb_root=dataset_root, categories=categories, videos=test_mp4_videos, video_folder="clips_mp4")
#assert train_ok and test_ok
print("All videos converted!")

# 4. Save .csv annotation file
train_df = prepare_csv(hmdb_root=dataset_root, categories=categories, videos=train_mp4_videos, video_folder="clips_mp4")
test_df = prepare_csv(hmdb_root=dataset_root, categories=categories, videos=test_mp4_videos, video_folder="clips_mp4")
train_df.to_csv(os.path.join(dataset_root, "train.csv"), header=False, index=False)
test_df.to_csv(os.path.join(dataset_root, "test.csv"), header=False, index=False)
print("Annotations saved!")

