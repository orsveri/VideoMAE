import io
import os
import math
import time
import json
import psutil
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
from torch.special import logit
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import subprocess
import torch
from torch import nn
import torch.distributed as dist
# from torch._six import inf
import math
inf = math.inf
import random

from tensorboardX import SummaryWriter


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'SLURM_PROCID' in os.environ:
        # args.rank = int(os.environ['SLURM_PROCID'])
        # args.gpu = int(os.environ['SLURM_LOCALID'])
        # args.world_size = int(os.environ['SLURM_NTASKS'])
        args.rank = int(os.environ['RANK'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)

        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {} | args.distributed={}'.format(
        args.rank, args.dist_url, args.gpu, args.distributed), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # assert torch.distributed.is_initialized()
    setup_for_distributed(args.rank == 0)
    #setup_for_distributed(True)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    extra_data = [item for sublist in extra_data for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data
    

def print_memory_usage():
    # GPU memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"------------GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # CPU memory usage
    vm = psutil.virtual_memory()
    used = (vm.total - vm.available) / (1024 ** 3)
    print(f"------------CPU Memory Usage: {used:.2f} GB / {vm.total / (1024 ** 3):.2f} GB")


def print_memory_usage():
    # GPU memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"------------GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # CPU memory usage
    vm = psutil.virtual_memory()
    used = (vm.total - vm.available) / (1024 ** 3)
    print(f"------------CPU Memory Usage: {used:.2f} GB / {vm.total / (1024 ** 3):.2f} GB")


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', multiplier=1.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.multiplier = multiplier

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss) # Probability for the true class
        focal_loss = self.multiplier * self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class FocalLoss2(nn.Module):
    def __init__(self, alpha=[0.40, 0.60], gamma=2, reduction='mean', multiplier=1.):
        super(FocalLoss2, self).__init__()
        self.alpha = alpha  # a list/array with a weight for each class
        self.gamma = gamma
        self.reduction = reduction
        self.multiplier = multiplier

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            at = torch.tensor(self.alpha, dtype=inputs.dtype, device=inputs.device)[targets]
            ce_loss = at * ce_loss

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss * self.multiplier

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class SmoothAPLoss(nn.Module):
    def __init__(self, delta=0.01):
        super(SmoothAPLoss, self).__init__()
        self.delta = delta

    def forward(self, predictions, labels):
        # Apply softmax and take the probability of the dangerous class (index 1)
        pred_probs = torch.nn.functional.softmax(predictions, dim=1)[:, 1]
        # Get positive and negative predictions
        positive_probs = pred_probs[labels == 1]
        negative_probs = pred_probs[labels == 0]

        # Sort scores for negative samples in ascending order
        sorted_neg_probs, _ = torch.sort(negative_probs)

        # Compute SmoothAP
        loss = 0.0
        for pos_prob in positive_probs:
            rank_diff = torch.relu(sorted_neg_probs - pos_prob + self.delta)
            loss += torch.sum(rank_diff)

        loss /= positive_probs.shape[0] if positive_probs.shape[0] > 0 else 1.0

        return loss


class TemporalExponentialLoss(nn.Module):
    def __init__(self, alpha_pre=0.1, alpha_post=0.5, max_time_pre=1.0, max_time_post=0.5):
        super(TemporalExponentialLoss, self).__init__()
        self.alpha_pre = alpha_pre
        self.alpha_post = alpha_post
        self.max_time_pre = max_time_pre
        self.max_time_post = max_time_post

    def forward(self, y_pred, y_true, time_to_anomaly):
        # Base cross-entropy loss
        base_loss = nn.functional.cross_entropy(y_pred, y_true, reduction='none')

        # Calculate exponential weights
        weight = torch.ones_like(y_true, dtype=torch.float)
        weight[time_to_anomaly < 0] = torch.exp(self.alpha_pre * time_to_anomaly[time_to_anomaly < 0])
        weight[time_to_anomaly > 0] = torch.exp(-self.alpha_post * time_to_anomaly[time_to_anomaly > 0])

        # Saturate weights outside transition window (set max weight to 1.0)
        weight = torch.clamp(weight, max=1.0)

        # Apply weights to the base loss
        weighted_loss = base_loss * weight

        return weighted_loss.mean()


def brier_score_loss(predictions, targets):
    """
    Computes the Brier score for binary classification with logits output.

    Arguments:
    predictions -- torch.Tensor of shape (batch_size, 2), containing logits for two classes.
    targets -- torch.Tensor of shape (batch_size,), containing ground truth labels (0 or 1).

    Returns:
    torch.Tensor: Brier score loss.
    """
    # Apply softmax to convert logits to probabilities for class "unsafe"
    probs = F.softmax(predictions, dim=1)[:, 1]

    # Create target probabilities (0 or 1)
    target_probs = targets.float()

    # Calculate the Brier score (mean squared error between predicted probs and true labels)
    brier_loss = torch.mean((probs - target_probs) ** 2)

    return brier_loss

def gather_predictions(preds, world_size, batch_size):
    # Step 1: Get the number of samples for the current rank
    num_samples = torch.tensor([len(preds)], device='cuda')
    all_num_samples = [torch.zeros(1, device='cuda', dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_num_samples, num_samples)
    all_num_samples = [ns * batch_size for ns in all_num_samples]

    # Step 2: Find the max number of samples across all ranks
    max_samples = torch.max(torch.cat(all_num_samples)).item()

    # Step 3: Pad current rank's predictions if needed
    flattened_preds = torch.cat(preds, dim=0)
    current_len = len(flattened_preds)
    if current_len < max_samples:
        padded_preds = torch.cat([flattened_preds, torch.zeros(max_samples - current_len, device='cuda')])
    else:
        padded_preds = flattened_preds

    # Step 4: Gather all the padded predictions from all ranks
    gathered_preds = [torch.zeros_like(padded_preds) for _ in range(world_size)]
    dist.all_gather(gathered_preds, padded_preds)

    # Step 5: Remove padding based on the original number of samples per rank
    all_preds = []
    for rank_idx, gathered_rank_preds in enumerate(gathered_preds):
        actual_len = all_num_samples[rank_idx].item()
        all_preds.append(gathered_rank_preds[:actual_len])

    all_preds = torch.cat(all_preds, dim=0)

    return all_preds

def gather_predictions_nontensor(info, world_size):
    # Step 1: Wrap the Python object in a list (if it's not already)
    if not isinstance(info, list):
        info = [info]

    # Step 2: Check if the list contains tensors
    if isinstance(info[0], torch.Tensor):
        # Move all tensors to CPU to make sure they can be gathered
        info = [tensor.cpu() for tensor in info]

    # Step 3: Gather the info object across all ranks using all_gather_object
    gathered_info = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_info, info)

    # Step 4: Flatten the list of gathered objects
    all_info = []
    for rank_info in gathered_info:
        all_info.extend(rank_info)

    return all_info


# class TemporalProximityLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.sigmoid_before = lambda x: 1 / (1 + torch.exp(-6 * (x + 1)))
#         self.sigmoid_after = lambda x: 1 / (1 + torch.exp(-12 * (-x + 0.5)))
#
#     def forward(self, logits, labels, time_to_anomaly):
#         """
#         logits: Tensor of shape (N, 2) - predicted logits for safe and anomalous classes
#         labels: Tensor of shape (N,) - binary labels (0 for safe, 1 for anomalous)
#         time_to_anomaly: Tensor of shape (N,) - time-to-anomaly values for each frame
#         """
#         probs_anomaly = torch.softmax(logits, dim=1)[:, 1]  # Probability of anomalous class
#
#         # Temporal proximity penalty
#         penalty = torch.zeros_like(labels, dtype=torch.float32)
#         penalty[time_to_anomaly > 0] = self.sigmoid_before(time_to_anomaly[time_to_anomaly > 0])
#         penalty[time_to_anomaly < 0] = self.sigmoid_after(time_to_anomaly[time_to_anomaly < 0])
#
#         # Combine penalties with anomaly probabilities
#         temporal_loss = penalty * (1 - labels) * probs_anomaly  # Penalize false positives
#         return temporal_loss.mean()

# It doesn't work
# class TemporalProximityLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
#         self.safe_sigmoid_before = lambda x: 1 / (1 + torch.exp(-6 * (x + 1)))
#         self.safe_sigmoid_after = lambda x: 1 / (1 + torch.exp(-12 * (-x + 0.5)))
#         self.anomaly_sigmoid_before = lambda x: 1 / (1 + torch.exp(6 * (x + 1)))
#         self.anomaly_sigmoid_after = lambda x: 1 / (1 + torch.exp(12 * (-x + 0.5)))
#
#
#     def forward(self, logits, labels, time_to_anomaly, t_A1, t_B1):
#         """
#         logits: Tensor of shape (N, 2) - predicted logits for safe and anomalous classes
#         labels: Tensor of shape (N,) - binary labels (0 for safe, 1 for anomalous)
#         time_to_anomaly: Tensor of shape (N,) - time-to-anomaly values for each frame
#         t_A, t_B: Scalars - start and end of anomaly range
#         t_A1, t_B1: Scalars - start of extended range before and after anomaly
#         """
#         probs = torch.softmax(logits, dim=1)  # Probabilities for both classes
#         probs_safe = probs[:, 0]  # Probability of safe class
#         probs_anomaly = probs[:, 1]  # Probability of anomalous class
#
#         # Step 1: Temporal penalties
#         penalty_safe = torch.zeros_like(labels, dtype=torch.float32)  # Default scaling factor
#         penalty_anomaly = torch.ones_like(labels, dtype=torch.float32)  # Default scaling factor
#
#         # Before anomaly range [t_A1, t_A]
#         before_mask = (time_to_anomaly >= t_A1) & (time_to_anomaly < 0)
#         after_mask = (time_to_anomaly > 0) & (time_to_anomaly <= t_B1)
#         penalty_safe[before_mask] = self.safe_sigmoid_before(time_to_anomaly[before_mask])
#         penalty_safe[after_mask] = self.safe_sigmoid_after(time_to_anomaly[after_mask])
#         penalty_anomaly[before_mask] = self.anomaly_sigmoid_before(time_to_anomaly[before_mask])
#         penalty_anomaly[after_mask] = self.anomaly_sigmoid_after(time_to_anomaly[after_mask])
#
#         # Step 2: Cross-entropy loss
#         ce_loss = self.cross_entropy(logits, labels)  # Per-sample cross-entropy loss
#         probs = nn.functional.softmax(logits, dim=-1)
#         loss_safe = np.abs(probs[0] - labels)
#         loss_anomaly = np.abs(probs[1] - labels)
#
#         # Step 3: Scale the loss
#         scaled_loss = ce_loss.clone()
#         # For safe class (labels == 0)
#         loss_safe[labels == 0] += penalty_safe[labels == 0]
#         # For anomalous class (labels == 1)
#         loss_anomaly[labels == 0] *= penalty_anomaly[labels == 0]
#         scaled_loss = loss_safe + loss_anomaly
#
#         # Step 4: Combine losses
#         return scaled_loss, penalty_anomaly, penalty_safe  #.mean()


class DoubleBCELoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', multiplier=1.):
        super(DoubleBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, smoothed_labels):
        """
        logits: Tensor of shape (N, 2) - output logits from the model for the two classes
        smoothed_labels: Tensor of shape (N, 2) - temporally smoothed labels for both neurons/classes

        Returns:
            Total loss (scalar): Sum of BCE losses for each class
            Individual losses: Tensor of shape (N, 2) - losses for each sample and each class
        """
        # Separate logits and labels for each neuron/class
        logits_safe, logits_anomaly = logits[:, 0], logits[:, 1]
        labels_safe, labels_anomaly = smoothed_labels[:, 0], smoothed_labels[:, 1]

        # Calculate BCE loss for each neuron
        loss_safe = self.bce_loss(logits_safe, labels_safe)
        loss_anomaly = self.bce_loss(logits_anomaly, labels_anomaly)

        # Combine losses
        total_loss = (loss_safe + loss_anomaly).mean()  # Sum losses for both classes, then take mean over all samples
        individual_losses = torch.stack([loss_safe, loss_anomaly], dim=1)  # Shape: (N, 2)

        # return total_loss, individual_losses
        return total_loss


def generate_smooth_labels(time_to_anomaly, t_A, t_B, t_A1, t_B1):
    """
    time_to_anomaly: Tensor of shape (N,) - time-to-anomaly values
    t_A, t_B: Scalars - start and end of anomaly range
    t_A1, t_B1: Scalars - start of extended range before and after anomaly
    """
    smooth_labels = torch.zeros_like(time_to_anomaly, dtype=torch.float32)

    # Anomaly range
    smooth_labels[(time_to_anomaly >= t_A) & (time_to_anomaly <= t_B)] = 1.0

    # Transition periods
    sigmoid_before = lambda x: 1 / (1 + torch.exp(-6 * (x + 1)))
    sigmoid_after = lambda x: 1 / (1 + torch.exp(-12 * (-x + 0.5)))

    smooth_labels[(time_to_anomaly > t_A1) & (time_to_anomaly < t_A)] = \
        sigmoid_before(time_to_anomaly[(time_to_anomaly > t_A1) & (time_to_anomaly < t_A)] - t_A)
    smooth_labels[(time_to_anomaly > t_B) & (time_to_anomaly < t_B1)] = \
        sigmoid_after(time_to_anomaly[(time_to_anomaly > t_B) & (time_to_anomaly < t_B1)] - t_B)

    return smooth_labels


def calculate_regression_metrics(probs_anomaly, smooth_labels):
    """
    probs_anomaly: Tensor of shape (N,) - predicted probabilities of anomalous class
    smooth_labels: Tensor of shape (N,) - temporally smoothed labels
    """
    mse = torch.mean((probs_anomaly - smooth_labels) ** 2)
    mae = torch.mean(torch.abs(probs_anomaly - smooth_labels))
    return mse, mae

