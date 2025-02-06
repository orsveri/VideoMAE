import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.function_base import disp
import zipfile
import torch
import decord
from PIL import Image
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from natsort import natsorted
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
import video_transforms as video_transforms 
import volume_transforms as volume_transforms

from bdd100k import VideoMAE_BDD100K
from dataset.sequencing import RegularSequencer, RegularSequencerWithStart


video_ext = (".mov", ".mp4", ".avi", ".mkv")


class VideoMAE_SHIFT(VideoMAE_BDD100K):

    def __init__(self, fps=10, target_fps=10, zipname="extracted_videos_10fps.zip", **kwargs):
        self.zipname = zipname
        super().__init__(fps=fps, target_fps=target_fps, **kwargs)
        print("attributes:")
    
    def _make_dataset_snellius(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting, "r") as split_f:
            clips = [line.strip() for line in split_f]
        # for iv in ignore_videos:
        #     assert iv in clips
        #     clips.remove(iv)
        assert len(clips) > 0, f"Cannot find any video clips for the given split: {setting}"
        return clips
    
    def _prepare_views(self):
        dataset_sequences = []
        N = len(self.clips)
        for i in tqdm(range(N), desc=f"Preparing views of len {self.new_length} with FPS {self.tfps}"):
            cn = os.path.splitext(self.clips[i])[0]
            subdirs, clip_dir = os.path.split(cn)
            # subfolder
            with zipfile.ZipFile(os.path.join(self.root, "data", subdirs, self.zipname), 'r') as zipf:
                framenames = [f for f in zipf.namelist() if os.path.dirname(f) == clip_dir and os.path.splitext(f)[1]==".jpg"]
            sequences = self.sequencer.get_sequences(timesteps_nb=len(framenames), input_frequency=self.fps)
            if sequences is None:
                print(f"No sequences found for clip: {cn}")
                continue
            dataset_sequences.extend([(i, seq) for seq in sequences])
        self.dataset_samples = dataset_sequences
    
    def _getitem_orig(self, index):
        sample = self.dataset_samples[index]
        
        images, _, __ = self.load_images(sample)  # T H W C
        assert len(images) > 0

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0
                                                                                                    ,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        return (process_data, mask)
    
    def _getitem_finetune_align(self, index):
        sample = self.dataset_samples[index]
        
        images, _, __ = self.load_images_cv2(sample)  # T H W C
        assert len(images) > 0

        # augment
        images = self._aug_frame(images, self.args)

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0
                                                                                                    ,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        return (process_data, mask)
    
    def load_images(self, dataset_sample, img_ext=".jpg"):
        clip_id, frame_seq = dataset_sample
        full_clip_name = self.clips[clip_id]
        subdirs, clip_dir = os.path.split(full_clip_name)
        img_names = [os.path.join(clip_dir, f"{idx:06d}.jpg") for idx in frame_seq]
        images = []
        with zipfile.ZipFile(os.path.join(self.root, "data", subdirs, self.zipname), 'r') as zipf:
            for fname in img_names:
                with zipf.open(fname) as file:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    print("Image doesn't exist! ", fname)
                    exit(1)
                # resze
                if self.intermediate_size is not None:
                    h, w, _ = img.shape
                    if h < w:
                        scale = self.intermediate_size / h
                        new_h, new_w = self.intermediate_size, int(w * scale)
                    else:
                        scale = self.intermediate_size / w
                        new_h, new_w = int(h * scale), self.intermediate_size
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                images.append(img)
        #view = np.stack(view, axis=0)
        return images, full_clip_name, img_names[-1]
    
    def load_images_cv2(self, dataset_sample, img_ext=".jpg"):
        clip_id, frame_seq = dataset_sample
        full_clip_name = self.clips[clip_id]
        subdirs, clip_dir = os.path.split(full_clip_name)
        img_names = [os.path.join(clip_dir, f"{idx:06d}.jpg") for idx in frame_seq]
        images = []
        with zipfile.ZipFile(os.path.join(self.root, "data", subdirs, self.zipname), 'r') as zipf:
            for fname in img_names:
                with zipf.open(fname) as file:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    print("Image doesn't exist! ", fname)
                    exit(1)
                # resze
                if self.intermediate_size is not None:
                    h, w, _ = img.shape
                    if h < w:
                        scale = self.intermediate_size / h
                        new_h, new_w = self.intermediate_size, int(w * scale)
                    else:
                        scale = self.intermediate_size / w
                        new_h, new_w = int(h * scale), self.intermediate_size
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #view = np.stack(view, axis=0)
        return images, full_clip_name, img_names[-1]
    
    def _aug_frame(
        self,
        buffer,
        args,
    ):
        h, w, _ = buffer[0].shape
        # first, resize to a bit larger size (e.g. 320) instead of the target one (224)
        min_side = min(h, w, self.intermediate_size)
        do_pad = video_transforms.pad_wide_clips(h, w, min_side)
        buffer = [do_pad(img) for img in buffer]
        if torch.rand(1).item() > 0.3:
            aug_transform = video_transforms.create_random_augment(
                input_size=(args.input_size, args.input_size),
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                do_transforms=video_transforms.DRIVE_TRANSFORMS
            )
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            buffer = aug_transform(buffer)
        else:
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        return buffer
    
    def __getitem__(self, index):
        return self._getitem(index)


class VideoMAE_SHIFT_prepared(VideoMAE_SHIFT):
    
    def __init__(self, clips_txt, views_txt, **kwargs):
        self.clips_txt = clips_txt
        self.views_txt = views_txt
        super().__init__(**kwargs)

    def _make_dataset_snellius(self, directory, setting):
        clips = []
        # read from the file
        with open(os.path.join(self.root, self.clips_txt), 'r') as file:
            clips = [line.rstrip() for line in file]
        assert len(clips) > 0, f"Cannot find any video clips for the given split: {setting}"
        return clips
    
    def _prepare_views(self):
        dataset_sequences = []
        # read from the file
        with open(os.path.join(self.root, self.views_txt), 'r') as file:
            for line in file:
                el1, el2 = line.strip().split(",", 1)
                el1 = int(el1.strip())
                el2 = [int(x.strip()) for x in el2.strip('[]').split(',')]
                dataset_sequences.append([el1, el2])
        self.dataset_samples = dataset_sequences


class MockArgs:
    def __init__(self):
        self.input_size = 224  # Example input size
        self.mask_type = 'tube'  # Masking type, 'tube' in this case
        self.window_size = (8, 14, 14)  # Example window size for TubeMaskingGenerator
        self.mask_ratio = 0.90  # Example mask ratio
        self.transforms_finetune_align = True


class CustomDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize attributes to store corrupt clips
        self.corrupt_clips = []
        self.corrupt_clips_decord = []


if __name__ == "__main__":
    from datasets import DataAugmentationForVideoMAE
    from multiprocessing import Manager
    manager = Manager()
    args = MockArgs()
    tf = DataAugmentationForVideoMAE(args)

    # create split file - trainval.txt
    if False:
        anno_path = "/gpfs/work3/0/tese0625/datasets/SHIFT/anno/_trainval.csv"
        rootdir = "/gpfs/work3/0/tese0625/datasets/SHIFT/data"
        rootdir_ = "gpfs/work3/0/tese0625/datasets/SHIFT"
        subdirs = (
            "train/front_continx1", "train/front_continx10", "train/front_continx100", 
            "train/front_discrete1", "train/front_discrete2", "train/front_discrete3", "train/front_discrete4", 
            "train/front_discrete5", "train/front_discrete6", "train/front_discrete7", "train/front_discrete8", 
            "train/front_discrete9", "train/front_discrete10",
            "val/front_continx1", "val/front_continx10", "val/front_continx100", "val/front_discrete"
        )
        out_anno_path = "/gpfs/work3/0/tese0625/datasets/SHIFT/anno/trainval2.txt"
        df = pd.read_csv(anno_path)
        videos = df["video"].tolist()
        del df
        # 
        dir_videos = []
        for sd in subdirs:
            with zipfile.ZipFile(os.path.join(rootdir, sd, "extracted_videos_10fps.zip"), 'r') as zipf:
                namelist = zipf.namelist()
            namelist = list(set([os.path.split(n)[0] for n in namelist]))
            dir_videos.append(namelist)

        for i in tqdm(range(len(videos))):
            clip_path = None
            for j, sd in enumerate(subdirs):
                if (videos[i]) in dir_videos[j]:
                    clip_path = sd
                    break
            assert clip_path is not None, f"Not found location: {videos[i]}"
            videos[i] = os.path.join(sd, videos[i])
        print("Res video 0", videos[0])
        with open(out_anno_path, "w") as f:
            f.write("\n".join(videos))
        exit(0)


    if True:
        dataset = VideoMAE_SHIFT(
        root="/gpfs/work3/0/tese0625/datasets/SHIFT",
        setting="/gpfs/work3/0/tese0625/datasets/SHIFT/anno/trainval.txt",
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=16,
        new_step=4,
        fps=10, 
        target_fps=10,
        transform=tf,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        args=args)
        L = len(dataset)
        print(f"\nLength of the dataset: {L}")

        clips = dataset.clips
        samples = dataset.dataset_samples

        os.makedirs("/gpfs/work3/0/tese0625/datasets/SHIFT/anno/prepared_views_16_fps10", exist_ok=True)

        print("Writing clips...")
        with open("/gpfs/work3/0/tese0625/datasets/SHIFT/anno/prepared_views_16_fps10/trainval_clips_step4.txt", "w") as file:
            for line in clips:
                file.write(line + "\n")
        print("\tClips done!")

        print("Writing samples...")
        with open("/gpfs/work3/0/tese0625/datasets/SHIFT/anno/prepared_views_16_fps10/trainval_dataset_samples_step4.txt", "w") as file:
            for s in samples:
                file.write(f"{s[0]},{s[1]}\n")
        print("\tSamples done!")

        print("Done!")

        exit(0)
    



