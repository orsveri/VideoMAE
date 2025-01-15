import os
import zipfile
import cv2
import numpy as np
import torch
import pandas as pd
import json
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import warnings
from torch.utils.data import Dataset

from functional import crop_clip
from random_erasing import RandomErasing
import video_transforms as video_transforms 
import volume_transforms as volume_transforms

from dataset.sequencing import RegularSequencer, RegularSequencerWithStart
from dataset.data_utils import smooth_labels, compute_time_vector


class FrameClsDataset_DoTA(Dataset):
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
        #self.new_height = new_height
        #self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.ttc_TT = args.ttc_TT if hasattr(args, "ttc_TT") else 2.
        self.ttc_TA = args.ttc_TA if hasattr(args, "ttc_TA") else 1.
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
        assert len(self._label_array) > 0

        if self.args.loss in ("2bce",):
            self.label_array = self._smoothed_label_array
        else:
            self.label_array = self._label_array

        count_safe = self._label_array.count(0)
        count_risk = self._label_array.count(1)
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
        clip_ttc = []
        clip_smoothed_labels = []

        with open(os.path.join(self.data_path, "dataset", self.anno_path), 'r') as file:
            clip_names = [line.rstrip() for line in file]
        for clip in clip_names:
            clip_anno_path = os.path.join(self.data_path, "dataset", "annotations", f"{clip}.json")
            with open(clip_anno_path) as f:
                anno = json.load(f)
                # sort is not required since we read already sorted timesteps from annotations
                timesteps = natsorted([int(os.path.splitext(os.path.basename(frame_label["image_path"]))[0]) for frame_label
                                  in anno["labels"]])
                cat_labels = [int(frame_label["accident_id"]) for frame_label in anno["labels"]]
                if_ego = anno["ego_involve"]
                if_night = anno["night"]
            binary_labels = [1 if l > 0 else 0 for l in cat_labels]
            ttc = compute_time_vector(binary_labels, fps=self.orig_fps, TT=self.ttc_TT, TA=self.ttc_TA)
            smoothed_labels = smooth_labels(labels=torch.Tensor(binary_labels), time_vector=ttc, before_limit=self.ttc_TT, after_limit=self.ttc_TA)

            clip_timesteps.append(timesteps)
            clip_binary_labels.append(binary_labels)
            clip_cat_labels.append(cat_labels)
            clip_ego.append(if_ego)
            clip_night.append(if_night)
            clip_ttc.append(ttc)
            clip_smoothed_labels.append(smoothed_labels)

        assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels) == len(clip_cat_labels)
        self.clip_names = clip_names
        self.clip_timesteps = clip_timesteps
        self.clip_bin_labels = clip_binary_labels
        self.clip_cat_labels = clip_cat_labels
        self.clip_ego = clip_ego
        self.clip_night = clip_night
        self.clip_ttc = clip_ttc
        self.clip_smoothed_labels = clip_smoothed_labels

    def _prepare_views(self):
        dataset_sequences = []
        label_array = []
        ttc = []
        smoothed_label_array = []
        sequencer = RegularSequencer(seq_frequency=self.target_fps, seq_length=self.view_len, step=self.view_step)
        N = len(self.clip_names)
        for i in range(N):
            timesteps = self.clip_timesteps[i]
            sequences = sequencer.get_sequences(timesteps_nb=len(timesteps), input_frequency=self.orig_fps)
            if sequences is None:
                continue
            dataset_sequences.extend([(i, seq) for seq in sequences])
            label_array.extend([self.clip_bin_labels[i][seq[-1]] for seq in sequences])
            smoothed_label_array.extend([self.clip_smoothed_labels[i][seq[-1]] for seq in sequences])
            ttc.extend([self.clip_ttc[i][seq[-1]] for seq in sequences])
        self.dataset_samples = dataset_sequences
        self._label_array = label_array
        self.ttc = ttc
        self._smoothed_label_array = smoothed_label_array

    def __getitem__(self, index):

        if self.mode == 'train':
            args = self.args
            sample = self.dataset_samples[index]
            buffer, _, __ = self.load_images(sample, final_resize=False, resize_scale=1.)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample, final_resize=False, resize_scale=1.)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                smoothed_label_list = []
                index_list = []
                ttc_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self._label_array[index]
                    smoothed_label = self._smoothed_label_array[index]
                    ttc = self.ttc[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    smoothed_label_list.append(smoothed_label)
                    index_list.append(index)
                    ttc_list.append(ttc)
                extra_info = [{"ttc": ttc_item, "smoothed_labels": slab_item} for ttc_item, slab_item in zip(ttc_list, smoothed_label_list)]
                return frame_list, label_list, index_list, extra_info
            else:
                buffer = self._aug_frame(buffer, args)
            extra_info = {"ttc": self.ttc[index], "smoothed_labels": self._smoothed_label_array[index]}
            return buffer, self._label_array[index], index, extra_info

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer, _, __ = self.load_images(sample, final_resize=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample, final_resize=True)
            buffer = self.data_transform(buffer)
            extra_info = {"ttc": self.ttc[index], "smoothed_labels": self._smoothed_label_array[index]}
            return buffer,self._label_array[index], index, extra_info

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            buffer, clip_name, frame_name = self.load_images(sample, final_resize=True)
            while len(buffer) == 0:
                warnings.warn("video {} not found during testing".format(str(self.test_dataset[index])))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                buffer, clip_name, frame_name = self.load_images(sample, final_resize=True)
            buffer = self.data_transform(buffer)
            extra_info = {"ttc": self.ttc[index], "clip": clip_name, "frame": frame_name, "smoothed_labels": self._smoothed_label_array[index]}
            return buffer, self._label_array[index], index, extra_info
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
        filenames = [f"{str(ts).zfill(6)}.jpg" for ts in timesteps]
        view = []
        with zipfile.ZipFile(os.path.join(self.data_path, "frames", clip_name, "images.zip"), 'r') as zipf:
            for fname in filenames:
                with zipf.open(fname) as file:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    print("Image doesn't exist! ", fname)
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
        return view, clip_name, filenames[-1]

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


# # before 1/(1+exp(-6*(x+1))), after 1/(1+exp(-12*(-x+0.5)))
# def compute_time_vector(labels, fps, TT=2, TA=1):
#     """
#     Compute time vector reflecting time in seconds before or after anomaly range.
#
#     Parameters:
#         labels (list or np.ndarray): Binary vector of frame labels (1 for anomalous, 0 otherwise).
#         fps (int): Frames per second of the video.
#         TT (float): Time-to-anomalous range in seconds (priority).
#         TA (float): Time-after-anomalous range in seconds.
#
#     Returns:
#         np.ndarray: Time vector for each frame.
#     """
#     num_frames = len(labels)
#     labels = np.array(labels)
#     default_value = max(TT, TA) * 2
#     time_vector = torch.zeros(num_frames, dtype=float)
#
#     # Get anomaly start and end indices
#     anomaly_indices = np.where(labels == 1)[0]
#     if len(anomaly_indices) == 0:
#         return time_vector  # No anomalies, return all zeros
#
#     # Define maximum frame thresholds for TT and TA
#     TT_frames = int(TT * fps)
#     TA_frames = int(TA * fps)
#
#     # Iterate through each frame
#     for i in range(num_frames):
#         if labels[i] == 1:
#             time_vector[i] = 0  # Anomalous frame, set to 0
#         else:
#             # Find distances to the start and end of anomaly ranges
#             distances_to_anomalies = anomaly_indices - i
#
#             # Time-to-closest-anomaly-range (TT priority)
#             closest_to_anomaly = distances_to_anomalies[distances_to_anomalies > 0]  # After the frame
#             if len(closest_to_anomaly) > 0 and closest_to_anomaly[0] <= TT_frames:
#                 time_vector[i] = -closest_to_anomaly[0] / fps
#                 continue
#
#             # Time-after-anomaly-range (TA range)
#             closest_after_anomaly = distances_to_anomalies[distances_to_anomalies < 0]  # Before the frame
#             if len(closest_after_anomaly) > 0 and abs(closest_after_anomaly[-1]) <= TA_frames:
#                 time_vector[i] = -closest_after_anomaly[-1] / fps
#                 continue
#
#             # Outside both TT and TA
#             time_vector[i] = -100.
#
#     return time_vector
#
#
# def smooth_labels(labels, time_vector, before_limit=2, after_limit=1):
#     xb = before_limit / 2
#     xa = after_limit / 2
#     kb = 12 / before_limit # 6 for before_limit=2
#     ka = 12 / after_limit # 12 for after_limit=1
#     sigmoid_before = lambda x: (1 / (1 + torch.exp(-kb * (x + xb)))).float()
#     sigmoid_after = lambda x: (1 / (1 + torch.exp(-ka * (-x + xa)))).float()
#
#     before_mask = (time_vector >= -before_limit) & (time_vector < 0)
#     after_mask = (time_vector > 0) & (time_vector <= after_limit)
#
#     target_anomaly = (labels == 1).float()
#     target_anomaly[before_mask] = sigmoid_before(time_vector[before_mask])
#     target_anomaly[after_mask] = sigmoid_after(time_vector[after_mask])
#     target_safe = 1 - target_anomaly
#     smoothed_target = torch.stack((target_safe, target_anomaly), dim=-1)
#     return smoothed_target


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


class VideoMAE_DoTA(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(self,
                 anno_path,
                 data_path,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 view_len=1,
                 view_step=1,
                 orig_fps=10,
                 target_fps=10,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False,
                 args=None
                 ):

        super(VideoMAE_DoTA, self).__init__()
        self.anno_path = anno_path
        self.data_path = data_path
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.view_len = view_len
        self.view_step = view_step
        self.ofps = orig_fps
        self.tfps = target_fps
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init
        self.ttc_TT = args.ttc_TT if hasattr(args, "ttc_TT") else 2.
        self.ttc_TA = args.ttc_TA if hasattr(args, "ttc_TA") else 1.
        # TODO: fill in parameters
        self.sequencer = RegularSequencerWithStart(seq_frequency=self.tfps, seq_length=self.view_len, step=self.view_step)

        if not self.lazy_init:
            self._read_anno()
            self._prepare_views()
            if len(self.dataset_samples) == 0:
                raise RuntimeError("Found 0 video clips in subfolders of: " + data_path)

    def _read_anno(self):
        clip_names = None
        clip_timesteps = []
        clip_binary_labels = []
        clip_cat_labels = []
        clip_ego = []
        clip_night = []
        clip_ttc = []
        clip_smoothed_labels = []

        with open(os.path.join(self.data_path, "dataset", self.anno_path), 'r') as file:
            clip_names = [line.rstrip() for line in file]
        for clip in clip_names:
            clip_anno_path = os.path.join(self.data_path, "dataset", "annotations", f"{clip}.json")
            with open(clip_anno_path) as f:
                anno = json.load(f)
                # sort is not required since we read already sorted timesteps from annotations
                timesteps = natsorted([int(os.path.splitext(os.path.basename(frame_label["image_path"]))[0]) for frame_label
                                  in anno["labels"]])
                cat_labels = [int(frame_label["accident_id"]) for frame_label in anno["labels"]]
                if_ego = anno["ego_involve"]
                if_night = anno["night"]
            binary_labels = [1 if l > 0 else 0 for l in cat_labels]
            ttc = compute_time_vector(binary_labels, fps=self.ofps, TT=self.ttc_TT, TA=self.ttc_TA)
            smoothed_labels = smooth_labels(labels=torch.Tensor(binary_labels), time_vector=ttc, before_limit=self.ttc_TT, after_limit=self.ttc_TA)

            clip_timesteps.append(timesteps)
            clip_binary_labels.append(binary_labels)
            clip_cat_labels.append(cat_labels)
            clip_ego.append(if_ego)
            clip_night.append(if_night)
            clip_ttc.append(ttc)
            clip_smoothed_labels.append(smoothed_labels)

        assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels) == len(clip_cat_labels)
        self.clip_names = clip_names
        self.clip_timesteps = clip_timesteps
        self.clip_bin_labels = clip_binary_labels
        self.clip_cat_labels = clip_cat_labels
        self.clip_ego = clip_ego
        self.clip_night = clip_night
        self.clip_ttc = clip_ttc
        self.clip_smoothed_labels = clip_smoothed_labels

    def _prepare_views(self):
        dataset_sequences = []
        label_array = []
        ttc = []
        smoothed_label_array = []
        sequencer = RegularSequencer(seq_frequency=self.tfps, seq_length=self.view_len, step=self.view_step)
        N = len(self.clip_names)
        for i in range(N):
            timesteps = self.clip_timesteps[i]
            sequences = sequencer.get_sequences(timesteps_nb=len(timesteps), input_frequency=self.ofps)
            if sequences is None:
                continue
            dataset_sequences.extend([(i, seq) for seq in sequences])
            label_array.extend([self.clip_bin_labels[i][seq[-1]] for seq in sequences])
            smoothed_label_array.extend([self.clip_smoothed_labels[i][seq[-1]] for seq in sequences])
            ttc.extend([self.clip_ttc[i][seq[-1]] for seq in sequences])
        self.dataset_samples = dataset_sequences
        self._label_array = label_array
        self.ttc = ttc
        self._smoothed_label_array = smoothed_label_array

    def load_images(self, dataset_sample, final_resize=False, resize_scale=None):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(6)}.jpg" for ts in timesteps]
        view = []
        with zipfile.ZipFile(os.path.join(self.data_path, "frames", clip_name, "images.zip"), 'r') as zipf:
            for fname in filenames:
                with zipf.open(fname) as file:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    print("Image doesn't exist! ", fname)
                    exit(1)
                img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                # if final_resize:
                #     img = cv2.resize(img, dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
                # elif resize_scale is not None:
                #     short_side = min(img.shape[:2])
                #     target_side = self.crop_size * resize_scale
                #     k = target_side / short_side
                #     img = cv2.resize(img, dsize=(0,0), fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
                # else:
                #     raise ValueError
                # Convert OpenCV image (numpy) to PIL.Image and append to view
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                view.append(img)
        #view = np.stack(view, axis=0)
        return view, clip_name, filenames[-1]

    def __getitem__(self, index):
        sample = self.dataset_samples[index]
        if self.video_loader:
            buffer, _, __ = self.load_images(sample, final_resize=False, resize_scale=1.)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample, final_resize=False, resize_scale=1.)

        process_data, mask = self.transform((buffer, None))  # T*C,H,W
        # T*C,H,W -> T,C,H,W -> C,T,H,W
        process_data = process_data.view((self.view_len, 3) + process_data.size()[-2:]).transpose(0, 1)
        return (process_data, mask)

    def __len__(self):
        return len(self.dataset_samples)

