import os

import cv2
import numpy as np
import torch
import pandas as pd
import json
from PIL import Image
from torchvision import transforms

from functional import crop_clip
from random_erasing import RandomErasing
import warnings
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms

from dataset.sequencing import RegularSequencer, UnsafeOverlapSequencer


class FrameClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train',
                 view_len=8, target_fps=10, orig_fps=10, view_step=10,
                 crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=1, test_num_crop=1, args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.view_len = view_len
        self.target_fps = target_fps
        self.orig_fps = orig_fps
        self.view_step = view_step
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        self._read_anno()
        self._prepare_views()
        assert len(self.dataset_samples) > 0
        assert len(self.label_array) > 0

        count_safe = self.label_array.count(0)
        count_risk = self.label_array.count(1)
        print(f"\n\n===\n[{mode}] | COUNT safe: {count_safe}\nCOUNT risk: {count_risk}\n===")

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(self.crop_size, self.crop_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = [(0, 0)]
            self.test_dataset = self.dataset_samples
            self.test_label_array = self.label_array

    def _read_anno(self):
        clip_names = None
        clip_timesteps = []
        clip_binary_labels = []
        clip_cat_labels = []
        clip_ego = []
        clip_night = []

        with open(os.path.join(self.data_path, "dataset", self.anno_path), 'r') as file:
            clip_names = [line.rstrip() for line in file]
        for clip in clip_names:
            clip_anno_path = os.path.join(self.data_path, "dataset", "annotations", f"{clip}.json")
            with open(clip_anno_path) as f:
                anno = json.load(f)
                timesteps = [int(os.path.splitext(os.path.basename(frame_label["image_path"]))[0]) for frame_label
                                  in anno["labels"]]
                cat_labels = [int(frame_label["accident_id"]) for frame_label in anno["labels"]]
                binary_labels = [1 if l > 0 else 0  for l in cat_labels]
                if_ego = anno["ego_involve"]
                if_night = anno["night"]
                clip_timesteps.append(timesteps)
                clip_binary_labels.append(binary_labels)
                clip_cat_labels.append(cat_labels)
                clip_ego.append(if_ego)
                clip_night.append(if_night)
        assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels) == len(clip_cat_labels)
        self.clip_names = clip_names
        self.clip_timesteps = clip_timesteps
        self.clip_bin_labels = clip_binary_labels
        self.clip_cat_labels = clip_cat_labels
        self.clip_ego = clip_ego
        self.clip_night = clip_night

    def _prepare_views(self):
        dataset_sequences = []
        label_array = []
        ttc = []
        sequencer = RegularSequencer(seq_frequency=self.target_fps, seq_length=self.view_len, step=self.view_step)
        N = len(self.clip_names)
        for i in range(N):
            timesteps = self.clip_timesteps[i]
            sequences = sequencer.get_sequences(timesteps_nb=len(timesteps), input_frequency=self.orig_fps)
            if sequences is None:
                continue
            dataset_sequences.extend([(i, seq) for seq in sequences])
            label_array.extend([self.clip_bin_labels[i][seq[-1]] for seq in sequences])
            ttc.extend([2.7 for seq in sequences])
        self.dataset_samples = dataset_sequences
        self.label_array = label_array
        self.ttc = ttc

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            sample = self.dataset_samples[index]
            buffer = self.load_images(sample, final_resize=False, resize_scale=1.)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_images(sample, final_resize=False, resize_scale=1.)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                ttc_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    ttc = self.ttc[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                    ttc_list.append(ttc)
                return frame_list, label_list, index_list, ttc_list
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, self.ttc[index]

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.load_images(sample, final_resize=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_images(sample, final_resize=True)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], index, self.ttc[index]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            buffer = self.load_images(sample, final_resize=True)
            while len(buffer) == 0:
                warnings.warn("video {} not found during testing".format(str(self.test_dataset[index])))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                buffer = self.load_images(sample, final_resize=True)
            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], index, self.ttc[index]
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        # Perform data augmentation - vertical padding and horizontal flip
        # add padding
        _PAD_MODES = ([None, None, None, None,
                       'black', 'black',
                       'color',
                       'reflect', 'reflect',
                       'replicate', 'replicate'])
        choice = torch.randint(0, len(_PAD_MODES), (1,)).item()
        padding_mode = _PAD_MODES[choice]
        if padding_mode is not None:
            padding_top = 0.
            padding_bottom = 0.
            h_top = int(buffer[0].shape[0] * padding_top)
            h_bot = int(buffer[0].shape[1] * padding_bottom)
            if padding_mode == "reflect":
                do_pad = lambda x: cv2.resize(
                    cv2.copyMakeBorder(x, h_top, h_bot, 0, 0, cv2.BORDER_REFLECT),
                    dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
            elif padding_mode == "replicate":
                do_pad = lambda x: cv2.resize(
                    cv2.copyMakeBorder(x, h_top, h_bot, 0, 0, cv2.BORDER_REPLICATE),
                    dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
            elif padding_mode in ('black', 'color'):
                color = torch.randint(0, 256, (3,)).tolist() if padding_mode == 'color' else [0, 0, 0]
                do_pad = lambda x: cv2.resize(
                    cv2.copyMakeBorder(x, h_top, h_bot, 0, 0, cv2.BORDER_CONSTANT, value=color),
                    dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
            else:
                raise ValueError
        else:
            do_pad = lambda x: cv2.resize(x, dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
        buffer = [do_pad(img) for img in buffer]

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def load_images(self, dataset_sample, final_resize=False, resize_scale=None):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [os.path.join(self.data_path, "frames", clip_name, "images", f"{str(ts).zfill(6)}.jpg") for ts in timesteps]
        view = []
        for fname in filenames:
            img = cv2.imread(fname)
            if img is None:
                print(fname)
                exit(1)
            if final_resize:
                img = cv2.resize(img, dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
            elif resize_scale is not None:
                short_side = min(img.shape[:2])
                target_side = self.crop_size * resize_scale
                k = target_side / short_side
                img = cv2.resize(img, dsize=(0,0), fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            else:
                raise ValueError
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
            view.append(img)
        #view = np.stack(view, axis=0)
        return view

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=1,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

