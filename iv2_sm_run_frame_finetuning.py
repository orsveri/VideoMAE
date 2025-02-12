import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

from InternVideo2_single_modality.datasets.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from InternVideo2_single_modality.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from datasets_frame import build_frame_dataset
import utils as utils1
from InternVideo2_single_modality.engines.iv2_engine_for_frame_finetuning import train_one_epoch, validation_one_epoch, final_test
from InternVideo2_single_modality.iv2_utils import NativeScalerWithGradNormCount as NativeScaler
from InternVideo2_single_modality.iv2_utils import multiple_samples_collate
from InternVideo2_single_modality import iv2_utils as utils
from InternVideo2_single_modality.models import *
from InternVideo2_single_modality.models.internvl_clip_vision import inflate_weight


def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--steps_per_print', default=1, type=int)
    parser.add_argument('--use_ceph_checkpoint', action='store_true',
                        help="whether use ceph to save and load checkpoint, may be some bug now")
    parser.set_defaults(use_ceph_checkpoint=False)
    parser.add_argument('--ceph_checkpoint_prefix', default='', type=str,
                        help='prefix for checkpoint in ceph')
    parser.add_argument('--ckpt_path_split', default='/exp/', type=str,
                        help='string for splitting the ckpt_path')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')
    parser.add_argument('--layer_scale_init_value', default=1e-5, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable LayerScale")
    parser.add_argument('--layerscale_no_force_fp32', action='store_true',
                        help="Not force fp32 for LayerScale")
    parser.set_defaults(layerscale_no_force_fp32=False)
    parser.add_argument('--sep_pos_embed', action='store_true',
                        help="whether use seperable position embedding")
    parser.add_argument('--center_init', action='store_true',
                        help="center initlization for patch embedding")

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--head_drop_path', type=float, default=0.0, metavar='PCT',
                        help='Head Drop path rate (default: 0.0)')
    parser.add_argument('--qkv_bias', action='store_true', help="whether use bias for qkv")

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--loss', default='crossentropy',
                        choices=['crossentropy', 'focal', 'focal6x100', 'focal2_6', 'focal2_2', 'smoothap', 'exponential1', "2bce"],
                        type=str, help='dataset')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m3-n3-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'), # ! orsveri
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.1)')  # ! orsveri
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=1)
    parser.add_argument('--test_num_crop', type=int, default=1)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--delete_head', action='store_true', help='whether delete head')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--checkpoint_num', default=0, type=int,
                        help='number of layers for using checkpoint')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--final_reduction', default='fc_norm', choices=['fc_norm', 'cls', 'none'],
                        type=str, help='type of reduction at the end of ViT encoder. cls: only CLS token, fc_norm: mean pooling, none: no reduction')
    parser.add_argument('--freeze_layers', default=None, type=str)

    # Dataset parameters
    parser.add_argument('--prefix', default='', type=str, help='prefix for data')
    parser.add_argument('--split', default=' ', type=str, help='split for metadata')
    parser.add_argument('--filename_tmpl', default='img_{:05}.jpg', type=str, help='file template')
    parser.add_argument('--data_path', default='you_data_path', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--use_decord', action='store_true',
                        help='whether use decord to load video, otherwise load image')
    parser.add_argument('--no_use_decord', action='store_false', dest='use_decord')
    parser.set_defaults(use_decord=False)
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=1)
    parser.add_argument('--sampling_rate_val', type=int, default=-1)
    parser.add_argument('--view_fps', type=int, default=10)  # DoTA, DADA2k only!
    parser.add_argument('--data_set', default='Kinetics-400', choices=['Kinetics-400', 'SSV2', 'UCF101', 'HMDB51', 'DoTA', 'DoTA_half', 'DADA2K', 'DADA2K_half','image_folder'],
                        type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--nb_samples_per_epoch', default=0, type=int)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--test_best', action='store_true',
                        help='Whether test the best model')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--bf16', default=False, action='store_true')
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')
    
    parser.add_argument('--eval_option', default='', type=str)

    known_args, _ = parser.parse_known_args()

    parser.local_rank = int(os.getenv("LOCAL_RANK", 0))

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    # EVAL ===============
    # experiment_dict = {
    # "1": ["baselines/bl1/lr1e3_b56x1_dsampl1val2_ld06_aam6n3", 16],
    # "2_14": ["baselines/bl1/dotaH_lr1e3_b28x2_dsampl1val2_ld06_aam6n3", 14],
    # "2_15": ["baselines/bl1/dotaH_lr1e3_b28x2_dsampl1val2_ld06_aam6n3", 15],
    # "3a_15": ["baselines/bl1/dada_lr5e4_b56x1_dsampl1val2_ld06_aam6n3", 15],
    # "3b_7": ["baselines/bl1/dada_lr1e3_b56x1_dsampl1val2_ld06_aam6n3", 7],
    # "3b_14": ["baselines/bl1/dada_lr1e3_b56x1_dsampl1val2_ld06_aam6n3", 14],
    # "4_9": ["baselines/bl1/dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3", 9],
    # "5_5": ["baselines/bl2/lr1e3_b56x1_dsampl1val2_ld06_aam6n3", 5],
    # "6_8": ["baselines/bl2/dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3", 8],
    # "7_4": ["baselines/bl2/dada_lr1e3_b56x1_dsampl1val2_ld06_aam6n3", 4],
    # "8_5": ["baselines/bl2/dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3", 5],
    # "13_14": ["ft_after_pretrain/pt_bdd/dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3", 14],
    # "14_15": ["ft_after_pretrain/pt_bdd/dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3", 15],
    # "15_8": ["ft_after_pretrain/pt_bdd/dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3", 8],
    # "16_8": ["ft_after_pretrain/pt_bdd/dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3", 8],
    # "16_13": ["ft_after_pretrain/pt_bdd/dadaH_lr1e3_b28x2_dsampl1val3_ld06_aam6n3", 13]
    # }
    experiment_dict = {
    "9_5": ["baselines/bl3/9_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3", 5],
    "10_6": ["baselines/bl3/dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3", 6],
    "10_18": ["baselines/bl3/dotah_lr1e3_b28x2_dsampl1val2_ld06_aam6n3", 18],
    "11": ["baselines/bl3/dada_lr1e3_b56x1_dsampl1val3_ld06_aam6n3", 3],
    "12": ["baselines/bl3/dadah_lr1e3_b28x2_dsampl1val3_ld06_aam6n3", 1],
    }
    exprmnt = str(args.eval_option)
    assert exprmnt in experiment_dict
    log_part, exp_ep = experiment_dict[exprmnt]
    base_log_dir = os.path.join("logs", log_part)
    out_dir = os.path.join(base_log_dir, f"eval_{args.data_set}_ckpt_{exp_ep}")
    args.finetune = os.path.join(base_log_dir, f"checkpoint-{exp_ep}.pth")
    args.log_dir = out_dir
    args.output_dir = out_dir
    assert os.path.exists(args.finetune)

    if args.eval:
        os.makedirs(args.output_dir, exist_ok=True)
    # =========================


    try:
        utils.init_distributed_mode(args)
        print("Distributed process initialized successfully.")
        print(f"\tRank {utils.get_rank()}, World Size: {utils.get_world_size()}, Device: {torch.cuda.current_device()}")
    except Exception as e:
        print(f"Initialization failed for rank {utils.get_rank()}: {e}")
        exit(1)

    if ds_init is not None:
        utils.create_internvideo2_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.enabled = True
    cudnn.benchmark = True

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    dataset_train = None
    if not args.eval:
        dataset_train, args.nb_classes = build_frame_dataset(is_train=True, test_mode=False, args=args)

        # sampler_train = torch.utils.data.DistributedSampler(
        #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=False
        # )
        # print("Sampler_train = %s" % str(sampler_train))

        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        if args.nb_samples_per_epoch and args.nb_samples_per_epoch < len(dataset_train):
            sampler_train = utils1.ShortDistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
                num_samples_per_epoch=args.nb_samples_per_epoch
            )
            num_training_steps_per_epoch = sampler_train.total_size // total_batch_size
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
            )
            num_training_steps_per_epoch = len(dataset_train) // total_batch_size
        print("Sampler_train = %s" % str(sampler_train))

    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_frame_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, _ = build_frame_dataset(is_train=False, test_mode=True, args=args)

    print(f"dset lengths: train {len(dataset_train) if dataset_train is not None else '<not used>'}, "
          f"val {len(dataset_val) if dataset_train is not None else '<not used>'}")

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # !!!
    if args.eval:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    if dataset_train is not None:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            prefetch_factor=4,  # orsveri
            collate_fn=collate_func,
        )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            #prefetch_factor=4  # orsveri
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            #prefetch_factor=4  # orsveri
        )
    else:
        data_loader_test = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        
    if hasattr(args, "qkv_bias"):
        print("QKV_bias: set")
    else:
        print("QKV_bias: No qkv bias attribute!")
        exit(0)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        num_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        sep_pos_embed=args.sep_pos_embed,
        fc_drop_rate=args.fc_drop_rate,
        drop_path_rate=args.drop_path,
        head_drop_path_rate=args.head_drop_path,
        use_checkpoint=args.use_checkpoint,
        checkpoint_num=args.checkpoint_num,
        init_scale=args.init_scale,
        init_values=args.layer_scale_init_value,
        layerscale_no_force_fp32=args.layerscale_no_force_fp32,
        qkv_bias=args.qkv_bias if hasattr(args, "qkv_bias") else False
    )

    exit(0)

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        if 'head.weight' in checkpoint_model.keys():
            if args.delete_head:
                print("Removing head from pretrained checkpoint")
                del checkpoint_model['head.weight']
                del checkpoint_model['head.bias']
            elif checkpoint_model['head.weight'].shape[0] == 710:
                if args.nb_classes == 400:
                    checkpoint_model['head.weight'] = checkpoint_model['head.weight'][:args.nb_classes]
                    checkpoint_model['head.bias'] = checkpoint_model['head.bias'][:args.nb_classes]
                elif args.nb_classes in [600, 700]:
                    # download from https://drive.google.com/drive/folders/17cJd2qopv-pEG8NSghPFjZo1UUZ6NLVm
                    map_path = f'./k710/label_mixto{args.nb_classes}.json'
                    print(f'Load label map from {map_path}')
                    with open(map_path) as f:
                        label_map = json.load(f)
                    checkpoint_model['head.weight'] = checkpoint_model['head.weight'][label_map]
                    checkpoint_model['head.bias'] = checkpoint_model['head.bias'][label_map]
                    
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.norm'):
                new_dict[key.replace("encoder.norm", "fc_norm")] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        if checkpoint_model['patch_embed.proj.weight'].shape[2] == 1 and model.patch_embed.tubelet_size > 1:
            print("Inflate patch embedding")
            print(f"Use center initilization: {args.center_init}")
            checkpoint_model['patch_embed.proj.weight'] = inflate_weight(
                checkpoint_model['patch_embed.proj.weight'][:, :, 0], 
                model.patch_embed.tubelet_size,
                center=args.center_init
            )

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # we use 8 frames for pretraining
            orig_t_size = 8
            new_t_size = args.num_frames * args.num_segments // model.patch_embed.tubelet_size
            # height (== width) for the checkpoint position embedding
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(orig_t_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (new_t_size))** 0.5)
            
            # class_token and dist_token are kept unchanged
            if orig_t_size != new_t_size:
                print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
                pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
                pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
                pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
                pos_embed_checkpoint = new_pos_embed

            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
        
        elif 'pos_embed_spatial' in checkpoint_model and 'pos_embed_temporal' in checkpoint_model:
            pos_embed_spatial_checkpoint = checkpoint_model['pos_embed_spatial']
            pos_embed_temporal_checkpoint = checkpoint_model['pos_embed_temporal']

            embedding_size = pos_embed_spatial_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 

            orig_t_size = pos_embed_temporal_checkpoint.shape[-2]
            new_t_size = args.num_frames // model.patch_embed.tubelet_size

            # height (== width) for the checkpoint position embedding
            orig_size = int(pos_embed_spatial_checkpoint.shape[-2] ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // new_t_size) ** 0.5)

            if orig_t_size != new_t_size:
                print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
                tmp_pos_embed = pos_embed_temporal_checkpoint.view(1, orig_t_size, -1, embedding_size)
                tmp_pos_embed = tmp_pos_embed.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
                tmp_pos_embed = torch.nn.functional.interpolate(tmp_pos_embed, size=new_t_size, mode='linear')
                tmp_pos_embed = tmp_pos_embed.view(1, -1, embedding_size, new_t_size)
                tmp_pos_embed = tmp_pos_embed.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
                checkpoint_model['pos_embed_temporal'] = tmp_pos_embed

            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                pos_tokens = pos_embed_spatial_checkpoint
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                checkpoint_model['pos_embed_spatial'] = pos_tokens

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    # # Freeze specific layers
    if args.freeze_layers is not None:
        if args.freeze_layers.startswith("first N blocks"):
            n_blocks = int(args.freeze_layers.split(";")[1])
            print(f"\nFreezing first N blocks: {n_blocks}")
            # Freeze first N blocks
            for name, param in model.named_parameters():
                if "blocks" in name:
                    layer_index = int(name.split(".")[1])
                    if layer_index < n_blocks:
                        # Skip freezing if it's a normalization or projection layer
                        if "norm" in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    else:
                        param.requires_grad = True
                elif "patch_embed" in name: 
                    param.requires_grad = False
                else:
                    # Keep other parameters outside "blocks" trainable
                    param.requires_grad = True

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    if not args.eval:
        #total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        #num_training_steps_per_epoch = len(dataset_train) // total_batch_size
        args.lr = args.lr * total_batch_size * args.num_sample / 256
        args.min_lr = args.min_lr * total_batch_size * args.num_sample / 256
        args.warmup_lr = args.warmup_lr * total_batch_size * args.num_sample / 256
        print("LR = %.8f" % args.lr)
        print("Total Batch size = %d" % total_batch_size)
        print("Repeated sample = %d" % args.num_sample)
        print("Update frequent = %d" % args.update_freq)
        print("Number of training examples = %d" % len(dataset_train))
        print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    if not args.eval:
        print("Use step level LR scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr, warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    with_ttc = False
    if args.loss == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == "focal":
        criterion = utils.FocalLoss(alpha=0.75, gamma=2)
    elif args.loss == "focal6x100":
        criterion = utils.FocalLoss(alpha=0.75, gamma=6, multiplier=100)
    elif args.loss == "focal2_6":
        criterion = utils.FocalLoss2(gamma=6, multiplier=50)
    elif args.loss == "focal2_2":
        criterion = utils.FocalLoss2(gamma=2, multiplier=10)
    elif args.loss == "2bce":
        criterion = utils.DoubleBCELoss()
    elif args.loss == "smoothap":
        criterion = utils.SmoothAPLoss()
    elif args.loss == "exponential1":
        criterion = utils.TemporalExponentialLoss(lambda_param=0.1)
        with_ttc = True
    else:
        raise NotImplementedError(f"Loss not implemented: {args.loss}")

    print("criterion = %s" % str(criterion))

    ceph_args = {
        'use_ceph_checkpoint': args.use_ceph_checkpoint,
        'ceph_checkpoint_prefix': args.ceph_checkpoint_prefix,
        'ckpt_path_split': args.ckpt_path_split,
        'local_rank': args.gpu,
    }
    if ceph_args['use_ceph_checkpoint']:
        print("Will automatically upload model on ceph")
        assert ceph_args['ceph_checkpoint_prefix'] != '', "Should set prefix for ceph checkpoint!"

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema,
        ceph_args=ceph_args,
    )
    
    print(f"Use bf16 {args.bf16}")

    if args.eval:
        os.makedirs(args.output_dir, exist_ok=True)
        preds_file = os.path.join(args.output_dir, f'predictions.csv')
        stats_file = os.path.join(args.output_dir, f'stats.txt')
        assert not os.path.exists(preds_file), "File already exists!"
        assert not os.path.exists(stats_file), "File already exists!"
        test_stats = final_test(data_loader_test, model, device, preds_file, stats_file,
                                plot_dir=os.path.join(args.output_dir, "plots"), 
                                ds=args.enable_deepspeed, bf16=args.bf16)
        torch.distributed.barrier()
        # if global_rank == 0:
        #     print("Start merging results...")
        #     final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
        #     print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        #     log_stats = {'Final top-1': final_top1,
        #                 'Final Top-5': final_top5}
        #     if args.output_dir and utils.is_main_process():
        #         with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #             f.write(json.dumps(log_stats) + "\n")
        exit(0)

    with open(os.path.join(args.output_dir, "params.json"), mode="w") as f:
        json.dump(vars(args), f, indent=2)
    grad_norm_dir = os.path.join(args.output_dir, "grad_norms")
    os.makedirs(grad_norm_dir, exist_ok=True)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_ap = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        print("\ttraining another epoch...")
        train_stats_, train_stats, plots, grad_norms = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            with_ttc=with_ttc, bf16=args.bf16
        )
        # save grad norms
        assert np.max(grad_norms["qkv"]) > 0., "grad_norms < 0!! "
        print(f"Epoch {epoch}, max grad_norms qkv: {np.max(grad_norms["qkv"]):.2f}")
        np.savez(os.path.join(grad_norm_dir, f"gradnorm_ep{epoch}.npz"), **grad_norms)

        if log_writer is not None:
            log_writer.update(train_acc=train_stats['metr_acc'], head="my_train", step=epoch)
            log_writer.update(train_ap=train_stats['ap'], head="my_train", step=epoch)
            log_writer.update(train_auroc=train_stats['auroc'], head="my_train", step=epoch)
            log_writer.update(train_recall=train_stats['recall'], head="my_train", step=epoch)
            log_writer.update(train_precision=train_stats['precision'], head="my_train", step=epoch)
            log_writer.update(train_f1=train_stats['f1'], head="my_train", step=epoch)
            #
            log_writer.update(train_logP_mean=train_stats['logitsP_mean'], head="my_train_extra", step=epoch)
            log_writer.update(train_logP_std=train_stats['logitsP_std'], head="my_train_extra", step=epoch)
            log_writer.update(train_logP_median=train_stats['logitsP_median'], head="my_train_extra", step=epoch)
            log_writer.update(train_logN_mean=train_stats['logitsN_mean'], head="my_train_extra", step=epoch)
            log_writer.update(train_logN_std=train_stats['logitsN_std'], head="my_train_extra", step=epoch)
            log_writer.update(train_logN_median=train_stats['logitsN_median'], head="my_train_extra", step=epoch)
            log_writer.update(train_probs_mean=train_stats['probs_mean'], head="my_train_extra", step=epoch)
            log_writer.update(train_probs_std=train_stats['probs_std'], head="my_train_extra", step=epoch)
            log_writer.update(train_probs_median=train_stats['probs_median'], head="my_train_extra", step=epoch)
            #
            [log_writer.writer.add_figure(f"train_plots/train_{k}", fig, global_step=epoch) for k, fig in plots.items()]
        # save last model with all the parameters so we can continue from it
        utils1.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, epoch_name="last"
                    )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils1.save_model_weights_only(
                    args=args, epoch=epoch, model_without_ddp=model_without_ddp)
        if data_loader_val is not None:
            test_stats_, test_stats, plots = validation_one_epoch(data_loader_val, model, device, with_ttc=with_ttc, ds=args.enable_deepspeed, bf16=args.bf16)
            print(f"Accuracy of the network on the {len(dataset_val)} val videos: {test_stats_['acc']:.1f}%")
            # if max_accuracy < test_stats["auroc"]:
            #     max_accuracy = test_stats["auroc"]
            #     if args.output_dir and args.save_ckpt:
            #         utils.save_model(
            #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #             loss_scaler=loss_scaler, epoch="bestauroc", model_ema=model_ema)
            # if max_ap < test_stats["ap"]:
            #     max_ap = test_stats["ap"]
            #     if args.output_dir and args.save_ckpt:
            #         utils.save_model(
            #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #             loss_scaler=loss_scaler, epoch="bestap", model_ema=model_ema)
            #
            # epoch_save_list = (1, 3, 4, 5, 7, 15)
            # if epoch in epoch_save_list:
            #     utils.save_model(
            #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #         loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema
            #     )
            #
            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(val_acc=test_stats_['acc'], head="val", step=epoch)
                log_writer.update(val_loss=test_stats_['loss'], head="val", step=epoch)
                log_writer.update(val_acc=test_stats['metr_acc'], head="val", step=epoch)
                log_writer.update(val_ap=test_stats['ap'], head="val", step=epoch)
                log_writer.update(val_auroc=test_stats['auroc'], head="val", step=epoch)
                log_writer.update(val_recall=test_stats['recall'], head="val", step=epoch)
                log_writer.update(val_precision=test_stats['precision'], head="val", step=epoch)
                log_writer.update(val_f1=test_stats['f1'], head="val", step=epoch)
                #
                log_writer.update(val_logP_mean=test_stats['logitsP_mean'], head="val_extra", step=epoch)
                log_writer.update(val_logP_std=test_stats['logitsP_std'], head="val_extra", step=epoch)
                log_writer.update(val_logP_median=test_stats['logitsP_median'], head="val_extra", step=epoch)
                log_writer.update(val_logN_mean=test_stats['logitsN_mean'], head="val_extra", step=epoch)
                log_writer.update(val_logN_std=test_stats['logitsN_std'], head="val_extra", step=epoch)
                log_writer.update(val_logN_median=test_stats['logitsN_median'], head="val_extra", step=epoch)
                log_writer.update(val_probs_mean=test_stats['probs_mean'], head="val_extra", step=epoch)
                log_writer.update(val_probs_std=test_stats['probs_std'], head="val_extra", step=epoch)
                log_writer.update(val_probs_median=test_stats['probs_median'], head="val_extra", step=epoch)
                #
                [log_writer.writer.add_figure(f"val_plots/val_{k}", fig, global_step=epoch) for k, fig in plots.items()]

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'train_{k}': v for k, v in train_stats_.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats_.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'train_{k}': v for k, v in train_stats_.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    if args.test_best:
        print("Auto testing the best model")
        print("Skip this - don't need it")
        # args.eval = True
        # utils.auto_load_model(
        #     args=args, model=model, model_without_ddp=model_without_ddp,
        #     optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema,
        #     ceph_args=ceph_args,
        # )
    test_stats = final_test(data_loader_test, model, device, preds_file, with_ttc=with_ttc, ds=args.enable_deepspeed, bf16=args.bf16)
    torch.distributed.barrier()
    # if global_rank == 0:
    #     print("Start merging results...")
    #     final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
    #     print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
    #     log_stats = {'Final top-1': final_top1,
    #                 'Final Top-5': final_top5}
    #     if args.output_dir and utils.is_main_process():
    #         with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #             f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
