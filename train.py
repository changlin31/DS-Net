#!/usr/bin/env python
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import logging
import random
from collections import OrderedDict
from datetime import datetime

import numpy as np
import yaml
from timm.utils import get_outdir, distribute_bn, update_summary
from timm.data import Dataset, create_loader, resolve_data_config, AugMixDataset
from timm.models import create_model, resume_checkpoint, convert_splitbn_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from dyn_slim.utils import model_profiling, setup_default_logging, CheckpointSaver, ModelEma
from dyn_slim.apis import train_epoch_slim, validate_slim, train_epoch_slim_gate

import torch
import torch.nn as nn

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP

    has_apex = False

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config',
                                                 add_help=False)
parser.add_argument('-c', '--config', default='config.yml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N',
                    help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=4,
                    metavar='N',
                    help='ratio of validation batch size to training batch size (default: 4)')
parser.add_argument('--drop', type=float, default=0.0, metavar='DROP',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.0, metavar='DROP',
                    help='Drop connect rate (default: 0.)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--min-step-lr', type=float, default=0, metavar='LR',
                    help='lower lr bound for step schedulers (0)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')
# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')
# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA amp for mixed precision training')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='prec1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "prec1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--slim_train', action='store_true', default=False)
parser.add_argument('--gate-train', action='store_true', default=False)
parser.add_argument('--ieb', action='store_true', default=False)
parser.add_argument('--test_mode', action='store_true', default=False)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    import os

    args, args_text = _parse_args()

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = 'train'
        if args.gate_train:
            exp_name += '-dynamic'
        if args.slim_train:
            exp_name += '-slimmable'
        exp_name += '-{}'.format(args.model)
        exp_info = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model])
        output_dir = get_outdir(output_base, exp_name, exp_info)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    setup_default_logging(outdir=output_dir, local_rank=args.local_rank)

    torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            logging.warning(
                'Using more than one GPU per process in distributed mode is not allowed. Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(backend='nccl',
        #                                      init_method='tcp://127.0.0.1:23334',
        #                                      rank=args.local_rank,
        #                                      world_size=int(os.environ['WORLD_SIZE']))
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        logging.info(
            'Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
            % (args.rank, args.world_size))
    else:
        logging.info('Training with a single process on %d GPUs.' % args.num_gpu)

    # --------- random seed -----------
    random.seed(args.seed)  # TODO: do we need same seed on all GPU?
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.manual_seed(args.seed + args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint)

    # optionally resume from a checkpoint
    resume_state = {}
    resume_epoch = None
    if args.resume:
        resume_state, resume_epoch = resume_checkpoint(model, args.resume)

    if args.local_rank == 0:
        logging.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model,
                                      verbose=args.local_rank == 0)

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    if args.num_gpu > 1:
        if args.amp:
            logging.warning(
                'AMP does not work well with nn.DataParallel, disabling. Use distributed mode for multi-GPU AMP.')
            args.amp = False
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model.cuda()

    if args.train_mode == 'se':
        optimizer = create_optimizer(args, model.get_se())
    elif args.train_mode == 'bn':
        optimizer = create_optimizer(args, model.get_bn())
    elif args.train_mode == 'all':
        optimizer = create_optimizer(args, model)
    elif args.train_mode == 'gate':
        optimizer = create_optimizer(args, model.get_gate())

    use_amp = False
    if has_apex and args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        use_amp = True
    if args.local_rank == 0:
        logging.info('NVIDIA APEX {}. AMP {}.'.format(
            'installed' if has_apex else 'not installed', 'on' if use_amp else 'off'))


    if resume_state and not args.no_resume_opt:
        # ----------- Load Optimizer ---------
        if 'optimizer' in resume_state:
            if args.local_rank == 0:
                logging.info('Restoring Optimizer state from checkpoint')
            optimizer.load_state_dict(resume_state['optimizer'])
        if use_amp and 'amp' in resume_state and 'load_state_dict' in amp.__dict__:
            if args.local_rank == 0:
                logging.info('Restoring NVIDIA AMP state from checkpoint')
            amp.load_state_dict(resume_state['amp'])
    del resume_state

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex:
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    logging.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                logging.error(
                    'Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex:
            model = DDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logging.info(
                    "Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
            model = DDP(model, device_ids=[
                args.local_rank],
                        find_unused_parameters=True)  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logging.info('Scheduled epochs: {}'.format(num_epochs))

    # ------------- data --------------
    train_dir = os.path.join(args.data, 'train')
    if not os.path.exists(train_dir):
        logging.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)
    collate_fn = None
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=args.train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
    )
    loader_bn = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=args.train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
    )

    eval_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(args.data, 'validation')
        if not os.path.isdir(eval_dir):
            logging.error('Validation folder does not exist at: {}'.format(eval_dir))
            exit(1)
    dataset_eval = Dataset(eval_dir)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # ------------- loss_fn --------------
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits,
                                        smoothing=args.smoothing).cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        validate_loss_fn = train_loss_fn
    if args.ieb:
        distill_loss_fn = SoftTargetCrossEntropy().cuda()
    else:
        distill_loss_fn = None

    if args.local_rank == 0:
        model_profiling(model, 224, 224, 1, 3,
                        use_cuda=True, verbose=True)
    else:
        model_profiling(model, 224, 224, 1, 3,
                        use_cuda=True, verbose=False)

    if not args.test_mode:
        # start training
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)
            train_metrics = OrderedDict([('loss', 0.)])
            # train
            if args.gate_train:
                train_metrics = train_epoch_slim_gate(
                    epoch, model, loader_train, optimizer, train_loss_fn, args,
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    use_amp=use_amp, model_ema=model_ema,
                    optimizer_step=args.optimizer_step)
            else:
                train_metrics = train_epoch_slim(
                    epoch, model, loader_train, optimizer,
                    loss_fn=train_loss_fn,
                    distill_loss_fn=distill_loss_fn,
                    args=args, lr_scheduler=lr_scheduler,
                    saver=saver, output_dir=output_dir,
                    use_amp=use_amp, model_ema=model_ema,
                    optimizer_step=args.optimizer_step,
                )
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    logging.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            # eval
            if args.gate_train:
                eval_sample_list = ['dynamic']
            else:
                if epoch % 10 == 0 and epoch != 0:
                    eval_sample_list = ['smallest', 'largest', 'uniform']
                else:
                    eval_sample_list = ['smallest', 'largest']

            eval_metrics = [validate_slim(model,
                                      loader_eval,
                                      validate_loss_fn,
                                      args,
                                      model_mode=model_mode)
                        for model_mode in eval_sample_list]

            if model_ema is not None and not args.model_ema_force_cpu:

                ema_eval_metrics = [validate_slim(model_ema.ema,
                                                  loader_eval,
                                                  validate_loss_fn,
                                                  args,
                                                  model_mode=model_mode)
                                    for model_mode in
                                    eval_sample_list]

                eval_metrics = ema_eval_metrics

            if isinstance(eval_metrics, list):
                eval_metrics = eval_metrics[0]

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            # save
            update_summary(
                epoch, train_metrics, eval_metrics,
                os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, args,
                    epoch=epoch, model_ema=model_ema, metric=save_metric, use_amp=use_amp)
        # end training
        if best_metric is not None:
            logging.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    # test
    eval_metrics = []
    for choice in range(args.num_choice):
        # reset bn if not smallest
        if choice != 0:
            for layer in model.modules():
                if isinstance(layer, nn.BatchNorm2d) or \
                        isinstance(layer, nn.SyncBatchNorm) or \
                        (has_apex and isinstance(layer, apex.parallel.SyncBatchNorm)):
                    layer.reset_running_stats()
            model.train()
            with torch.no_grad():
                for batch_idx, (input, target) in enumerate(loader_bn):
                    if args.slim_train:
                        if hasattr(model, 'module'):
                            model.module.set_mode('uniform', choice=choice)
                        else:
                            model.set_mode('uniform', choice=choice)
                        model(input)

                    if batch_idx % 1000 == 0 and batch_idx != 0:
                        print('Subnet {} : reset bn for {} steps'.format(choice, batch_idx))
                        break
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    logging.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

        eval_metrics.append(validate_slim(model,
                                          loader_eval,
                                          validate_loss_fn,
                                          args,
                                          model_mode=choice))
    if args.local_rank == 0:
        print('Test results of the last epoch:\n', eval_metrics)


if __name__ == '__main__':
    main()
