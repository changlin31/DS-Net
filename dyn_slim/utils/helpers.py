import glob
import logging
import math
import operator
import os
import shutil
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn

try:
    from apex import amp

    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from dyn_slim.models.dyn_slim_ops import *


def unwrap_model(model):
    if isinstance(model, ModelEma):
        return unwrap_model(model.ema)
    else:
        return model.module if hasattr(model, 'module') else model


def get_state_dict(model):
    return unwrap_model(model).state_dict()

def setup_default_logging(outdir, local_rank, level='INFO'):
    log_format = '%(asctime)s %(message)s'
    datefmt = '%m/%d %I:%M:%S %p'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(log_format, datefmt))
    sh.setLevel(eval('logging.'+level))
    logger.addHandler(sh)
    if local_rank == 0:
        fh = logging.FileHandler(os.path.join(outdir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format, datefmt))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)


class CheckpointSaver:
    def __init__(
            self,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10):

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        assert self.max_history >= 1

    def save_checkpoint(self, model, optimizer, args, epoch, model_ema=None, metric=None,
                        use_amp=False):
        assert epoch >= 0
        # worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        # if (len(self.checkpoint_files) < self.max_history
        #         or metric is None):
        if len(self.checkpoint_files) >= self.max_history:
            self._cleanup_checkpoints(1)
        filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
        save_path = os.path.join(self.checkpoint_dir, filename)
        self._save(save_path, model, optimizer, args, epoch, model_ema, metric,
                   use_amp)
        self.checkpoint_files.append((save_path, metric))
        self.checkpoint_files = sorted(
            self.checkpoint_files, key=lambda x: x[1],
            reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

        checkpoints_str = "Current checkpoints:\n"
        for c in self.checkpoint_files:
            checkpoints_str += ' {}\n'.format(c)
        logging.info(checkpoints_str)

        if metric is not None and (
                self.best_metric is None or self.cmp(metric, self.best_metric)):
            self.best_epoch = epoch
            self.best_metric = metric
            shutil.copyfile(save_path, os.path.join(self.checkpoint_dir,
                                                    'model_best' + self.extension))

        return (None, None) if self.best_metric is None else (
            self.best_metric, self.best_epoch)

    def _save(self, save_path, model, optimizer, args, epoch, model_ema=None, metric=None,
              use_amp=False):
        save_state = {
            'epoch': epoch,
            'arch': args.model,
            'state_dict': get_state_dict(model),
            'optimizer': optimizer.state_dict(),
            'args': args,
            'version': 2,  # version < 2 increments epoch before save
        }
        if use_amp and 'state_dict' in amp.__dict__:
            save_state['amp'] = amp.state_dict()
        if model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(model_ema)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index <= 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                logging.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                logging.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, model, optimizer, args, epoch, model_ema=None, use_amp=False,
                      batch_idx=0):
        assert epoch >= 0
        filename = '-'.join(
            [self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, model, optimizer, args, epoch, model_ema, use_amp=use_amp)
        if os.path.exists(self.last_recovery_file):
            try:
                logging.debug("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                logging.error(
                    "Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        if len(files):
            return files[0]
        else:
            return ''


def _init_weight_goog(m, n='', fix_group_fanout=True):
    """ Weight initialization as per Tensorflow official implementations.

    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct fanout calculation w/ group convs

    FIXME change fix_group_fanout to default to True if experiments show better training results

    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.affine:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        if m.bias is not None:
            m.bias.data.zero_()


def efficientnet_init_weights(model: nn.Module, init_fn=None):
    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n)


def accuracy(output, target, topk=(1,), no_reduce=False):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    conf, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if no_reduce:
        return conf, correct
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True, strict=True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                logging.info('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=strict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    logging.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    logging.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                logging.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


class ModelEma:
    """ Model Exponential Moving Average (DEPRECATED)

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This version is deprecated, it does not work with scripted models. Will be removed eventually.

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device='', resume='', log_info=True, resume_strict=True):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        self.log_info = log_info
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self.strict = resume_strict
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict, strict=self.strict)
            if self.log_info:
                logging.info("Loaded state_dict_ema")
        else:
            logging.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
