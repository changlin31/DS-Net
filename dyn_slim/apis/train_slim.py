import logging
import os
import random
import time
from collections import OrderedDict

import numpy as np

from dyn_slim.apis.train_slim_gate import add_mac_hooks, print_gate_stats
from dyn_slim.utils import add_flops

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP

    has_apex = False

from timm.utils import *

import torch
import torch.nn.functional as F
import torchvision.utils


def train_epoch_slim(
        epoch, model, loader, optimizer, loss_fn, distill_loss_fn, args, lr_scheduler=None,
        saver=None, output_dir='', use_amp=False, model_ema=None, optimizer_step=1):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m_largest = AverageMeter()
    losses_m_smallest = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    seed = num_updates
    loss_largest = torch.zeros(1).cuda()
    loss_smallest = torch.zeros(1).cuda()
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()

        sample_list = ['largest', 'uniform', 'uniform', 'smallest']
        guide_list = []

        for sample_idx, model_mode in enumerate(sample_list):
            seed = seed + 1
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if hasattr(model, 'module'):
                model.module.set_mode(model_mode)
            else:
                model.set_mode(model_mode)

            output = model(input)

            if model_mode == 'largest':
                loss = loss_fn(output, target)
                if args.ieb:
                    with torch.no_grad():
                        if hasattr(model_ema.ema, 'module'):
                            model_ema.ema.module.set_mode(model_mode)
                        else:
                            model_ema.ema.set_mode(model_mode)
                        output_largest = model_ema.ema(input)
                    guide_list.append(output_largest)
                loss_largest = loss
            elif model_mode != 'smallest':
                if args.ieb:
                    loss = distill_loss_fn(output, F.softmax(output_largest, dim=1))
                    # with torch.no_grad():
                    #     guide_output = model_ema.ema(input)
                    with torch.no_grad():
                        if hasattr(model_ema.ema, 'module'):
                            model_ema.ema.module.set_mode(model_mode)
                        else:
                            model_ema.ema.set_mode(model_mode)
                        guide_output = model_ema.ema(input)
                    guide_list.append((guide_output))
                    # guide_list.append((output.detach()))
                else:
                    loss = loss_fn(output, target)
            else:  # 'smallest'
                soft_labels_ = [torch.unsqueeze(guide_list[idx], dim=2) for
                                idx in range(len(guide_list))]
                soft_labels_softmax = [F.softmax(i, dim=1) for i in soft_labels_]
                soft_labels_softmax = torch.cat(soft_labels_softmax, dim=2).mean(dim=2)
                if args.ieb:
                    loss = distill_loss_fn(output, soft_labels_softmax)
                else:
                    loss = loss_fn(output, target)
                loss_smallest = loss

            loss = loss / optimizer_step

            if use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        if last_batch or (batch_idx + 1) % optimizer_step == 0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    if not p.grad is None and torch.sum(torch.abs(p.grad.data)) == 0.0:
                        p.grad = None
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss_largest = reduce_tensor(loss_largest.data, args.world_size)
                reduced_loss_smallest = reduce_tensor(loss_smallest.data, args.world_size)
            else:
                reduced_loss_largest = loss_largest.data
                reduced_loss_smallest = loss_smallest.data

            losses_m_largest.update(reduced_loss_largest.item(), input.size(0))
            losses_m_smallest.update(reduced_loss_smallest.item(), input.size(0))

            if args.local_rank == 0:
                logging.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss_smallest{isdistill}: {loss_smallest.val:>9.6f} ({loss_smallest.avg:>6.4f})  '
                    'Loss_largest: {loss_largest.val:>9.6f} ({loss_largest.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, last_idx,
                        100. * batch_idx / last_idx,
                        isdistill='(distill)' if args.ieb else '',
                        loss_smallest=losses_m_smallest,
                        loss_largest=losses_m_largest,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                model, optimizer, args, epoch, model_ema=model_ema, use_amp=use_amp,
                batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m_largest.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss_largest', losses_m_largest.avg),
                        ('loss_smallest', losses_m_smallest.avg)])


def validate_slim(model, loader, loss_fn, args, log_suffix='', model_mode='largest'):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    flops_m = AverageMeter()

    model.eval()
    model.apply(lambda m: add_mac_hooks(m))
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if not isinstance(model_mode, str):
                if hasattr(model, 'module'):
                    model.module.set_mode('uniform', choice=model_mode)
                else:
                    model.set_mode('uniform', choice=model_mode)
            else:
                if hasattr(model, 'module'):
                    model.module.set_mode(model_mode)
                else:
                    model.set_mode(model_mode)
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            running_flops = add_flops(model)
            if isinstance(running_flops, torch.Tensor):
                running_flops = running_flops.float().mean().cuda()
            else:
                running_flops = torch.FloatTensor([running_flops]).cuda()

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                prec1 = reduce_tensor(prec1, args.world_size)
                prec5 = reduce_tensor(prec5, args.world_size)
                running_flops = reduce_tensor(running_flops, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))
            flops_m.update(running_flops.item(), input.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and last_batch:
                if model_mode == 'dynamic':
                    print_gate_stats(model)
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Mode: {mode}   '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                    'Flops: {flops.val:>6.0f} ({flops.avg:>6.0f})'.format(
                        log_name, batch_idx, last_idx,
                        batch_time=batch_time_m,
                        mode=model_mode,
                        loss=losses_m,
                        flops=flops_m,
                        top1=prec1_m, top5=prec5_m))

    metrics = OrderedDict(
        [('loss', losses_m.avg), ('prec1', prec1_m.avg), ('prec5', prec5_m.avg), ('flops', flops_m.avg)])

    return metrics
