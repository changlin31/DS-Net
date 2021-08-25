import logging
import time
from collections import OrderedDict

from dyn_slim.models.dyn_slim_blocks import MultiHeadGate
from dyn_slim.models.dyn_slim_ops import DSBatchNorm2d
from dyn_slim.utils import add_flops, accuracy

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP

    has_apex = False
from timm.utils import AverageMeter, reduce_tensor

import numpy as np
import torch
import torch.nn as nn

model_mac_hooks = []


def train_epoch_slim_gate(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False, model_ema=None,
        optimizer_step=1):
    start_chn_idx = args.start_chn_idx
    num_gate = 1

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    acc_m = AverageMeter()
    flops_m = AverageMeter()
    ce_loss_m = AverageMeter()
    flops_loss_m = AverageMeter()
    acc_gate_m_l = [AverageMeter() for i in range(num_gate)]
    gate_loss_m_l = [AverageMeter() for i in range(num_gate)]
    model.train()
    for n, m in model.named_modules():  # Freeze bn
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, DSBatchNorm2d):
            m.eval()

    for n, m in model.named_modules():
        if len(getattr(m, 'in_channels_list', [])) > 4:
            m.in_channels_list = m.in_channels_list[start_chn_idx:4]
            m.in_channels_list_tensor = torch.from_numpy(
                np.array(m.in_channels_list)).float().cuda()
        if len(getattr(m, 'out_channels_list', [])) > 4:
            m.out_channels_list = m.out_channels_list[start_chn_idx:4]
            m.out_channels_list_tensor = torch.from_numpy(
                np.array(m.out_channels_list)).float().cuda()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    model.apply(lambda m: add_mac_hooks(m))
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()

        if last_batch or (batch_idx + 1) % optimizer_step == 0:
            optimizer.zero_grad()
        # generate online labels
        with torch.no_grad():
            set_model_mode(model, 'smallest')
            output = model(input)
            conf_s, correct_s = accuracy(output, target, no_reduce=True)
            gate_target = [torch.LongTensor([0]) if correct_s[0][idx] else torch.LongTensor([3])
                           for idx in range(correct_s[0].size(0))]
            gate_target = torch.stack(gate_target).squeeze(-1).cuda()
        # =============
        set_model_mode(model, 'dynamic')
        output = model(input)

        if hasattr(model, 'module'):
            model_ = model.module
        else:
            model_ = model

        #  SGS Loss
        gate_loss = 0
        gate_num = 0
        gate_loss_l = []
        gate_acc_l = []
        for n, m in model_.named_modules():
            if isinstance(m, MultiHeadGate):
                if getattr(m, 'keep_gate', None) is not None:
                    gate_num += 1
                    g_loss = loss_fn(m.keep_gate, gate_target)
                    gate_loss += g_loss
                    gate_loss_l.append(g_loss)
                    gate_acc_l.append(accuracy(m.keep_gate, gate_target, topk=(1,))[0])

        gate_loss /= gate_num

        #  MAdds Loss
        running_flops = add_flops(model)
        if isinstance(running_flops, torch.Tensor):
            running_flops = running_flops.float().mean().cuda()
        else:
            running_flops = torch.FloatTensor([running_flops]).cuda()
        flops_loss = (running_flops / 1e9) ** 2

        #  Target Loss, back-propagate through gumbel-softmax
        ce_loss = loss_fn(output, target)

        loss = gate_loss + ce_loss + 0.5 * flops_loss
        # loss = ce_loss
        acc1 = accuracy(output, target, topk=(1,))[0]

        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if last_batch or (batch_idx + 1) % optimizer_step == 0:
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            acc_m.update(acc1.item(), input.size(0))
            flops_m.update(running_flops.item(), input.size(0))
            ce_loss_m.update(ce_loss.item(), input.size(0))
            flops_loss_m.update(flops_loss.item(), input.size(0))
        else:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            reduced_acc = reduce_tensor(acc1, args.world_size)
            reduced_flops = reduce_tensor(running_flops, args.world_size)
            reduced_loss_flops = reduce_tensor(flops_loss, args.world_size)
            reduced_ce_loss = reduce_tensor(ce_loss, args.world_size)
            reduced_acc_gate_l = reduce_list_tensor(gate_acc_l, args.world_size)
            reduced_gate_loss_l = reduce_list_tensor(gate_loss_l, args.world_size)
            losses_m.update(reduced_loss.item(), input.size(0))
            acc_m.update(reduced_acc.item(), input.size(0))
            flops_m.update(reduced_flops.item(), input.size(0))
            flops_loss_m.update(reduced_loss_flops.item(), input.size(0))
            ce_loss_m.update(reduced_ce_loss.item(), input.size(0))
            for i in range(num_gate):
                acc_gate_m_l[i].update(reduced_acc_gate_l[i].item(), input.size(0))
                gate_loss_m_l[i].update(reduced_gate_loss_l[i].item(), input.size(0))
        batch_time_m.update(time.time() - end)
        if (last_batch or batch_idx % args.log_interval == 0) and args.local_rank == 0 and batch_idx != 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            print_gate_stats(model)
            logging.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'CELoss: {celoss.val:>9.6f} ({celoss.avg:>6.4f})  '
                'GateLoss: {gate_loss[0].val:>6.4f} ({gate_loss[0].avg:>6.4f})  '
                'FlopsLoss: {flopsloss.val:>9.6f} ({flopsloss.avg:>6.4f})  '
                'TrainAcc: {acc.val:>9.6f} ({acc.avg:>6.4f})  '
                'GateAcc: {acc_gate[0].val:>6.4f}({acc_gate[0].avg:>6.4f})  '
                'Flops: {flops.val:>6.0f} ({flops.avg:>6.0f})  '
                'LR: {lr:.3e}  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'DataTime: {data_time.val:.3f} ({data_time.avg:.3f})\n'.format(
                    epoch,
                    batch_idx, last_idx,
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    flopsloss=flops_loss_m,
                    acc=acc_m,
                    flops=flops_m,
                    celoss=ce_loss_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) * args.world_size / batch_time_m.val,
                    rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m,
                    gate_loss=gate_loss_m_l,
                    acc_gate=acc_gate_m_l
                )
            )

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                model, optimizer, args, epoch, model_ema=model_ema, use_amp=use_amp,
                batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


@torch.no_grad()
def validate_gate(model, loader, loss_fn, args, log_suffix=''):
    start_chn_idx = args.start_chn_idx
    num_gate = 1

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    flops_m = AverageMeter()
    acc_gate_m_l = [AverageMeter() for i in range(num_gate)]
    model.eval()

    for n, m in model.named_modules():
        if len(getattr(m, 'in_channels_list', [])) > 4:
            m.in_channels_list = m.in_channels_list[start_chn_idx:4]
            m.in_channels_list_tensor = torch.from_numpy(
                np.array(m.in_channels_list)).float().cuda()
        if len(getattr(m, 'out_channels_list', [])) > 4:
            m.out_channels_list = m.out_channels_list[start_chn_idx:4]
            m.out_channels_list_tensor = torch.from_numpy(
                np.array(m.out_channels_list)).float().cuda()

    end = time.time()
    last_idx = len(loader) - 1
    model.apply(lambda m: add_mac_hooks(m))
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
        # generate online labels
        with torch.no_grad():
            set_model_mode(model, 'smallest')
            output = model(input)
            conf_s, correct_s = accuracy(output, target, no_reduce=True)
            gate_target = [torch.LongTensor([0]) if correct_s[0][idx] else torch.LongTensor([3])
                           for idx in range(correct_s[0].size(0))]
            gate_target = torch.stack(gate_target).squeeze(-1).cuda()
        # =============
        set_model_mode(model, 'dynamic')
        output = model(input)

        if hasattr(model, 'module'):
            model_ = model.module
        else:
            model_ = model

        gate_acc_l = []
        for n, m in model_.named_modules():
            if isinstance(m, MultiHeadGate):
                if getattr(m, 'keep_gate', None) is not None:
                    gate_acc_l.append(accuracy(m.keep_gate, gate_target, topk=(1,))[0])

        running_flops = add_flops(model)
        if isinstance(running_flops, torch.Tensor):
            running_flops = running_flops.float().mean().cuda()
        else:
            running_flops = torch.FloatTensor([running_flops]).cuda()

        loss = loss_fn(output, target)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            prec1_m.update(prec1.item(), input.size(0))
            prec5_m.update(prec5.item(), input.size(0))
            flops_m.update(running_flops.item(), input.size(0))
        else:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            reduced_prec1 = reduce_tensor(prec1, args.world_size)
            reduced_prec5 = reduce_tensor(prec5, args.world_size)
            reduced_flops = reduce_tensor(running_flops, args.world_size)
            reduced_acc_gate_l = reduce_list_tensor(gate_acc_l, args.world_size)
            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            prec1_m.update(reduced_prec1.item(), input.size(0))
            prec5_m.update(reduced_prec5.item(), input.size(0))
            flops_m.update(reduced_flops.item(), input.size(0))
            for i in range(num_gate):
                acc_gate_m_l[i].update(reduced_acc_gate_l[i].item(), input.size(0))
        batch_time_m.update(time.time() - end)
        if (last_batch or batch_idx % args.log_interval == 0) and args.local_rank == 0 and batch_idx != 0:
            print_gate_stats(model)
            log_name = 'Test' + log_suffix
            logging.info(
                '{}: [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                'Acc@1: {prec1.val:>9.6f} ({prec1.avg:>6.4f})  '
                'Acc@5: {prec5.val:>9.6f} ({prec5.avg:>6.4f})  '
                'GateAcc: {acc_gate[0].val:>6.4f}({acc_gate[0].avg:>6.4f})  '
                'Flops: {flops.val:>6.0f} ({flops.avg:>6.0f})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'DataTime: {data_time.val:.3f} ({data_time.avg:.3f})\n'.format(
                    log_name,
                    batch_idx, last_idx,
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    prec1=prec1_m,
                    prec5=prec5_m,
                    flops=flops_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) * args.world_size / batch_time_m.val,
                    rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                    data_time=data_time_m,
                    acc_gate=acc_gate_m_l
                )
            )

        end = time.time()
        # end for
    metrics = OrderedDict(
        [('loss', losses_m.avg), ('prec1', prec1_m.avg), ('prec5', prec5_m.avg), ('flops', flops_m.avg)])

    return metrics


def reduce_list_tensor(tensor_l, world_size):
    ret_l = []
    for tensor in tensor_l:
        ret_l.append(reduce_tensor(tensor, world_size))
    return ret_l


def set_gate(m, gate=None):
    if gate is not None:
        gate = gate.cuda()
    if hasattr(m, 'gate'):
        setattr(m, 'gate', gate)


def module_mac(self, input, output):
    if isinstance(input[0], tuple):
        if isinstance(input[0][0], list):
            ins = input[0][0][3].size()
        else:
            ins = input[0][0].size()
    else:
        ins = input[0].size()
    if isinstance(output, tuple):
        if isinstance(output[0], list):
            outs = output[0][3].size()
        else:
            outs = output[0].size()
    else:
        outs = output.size()
    if isinstance(self, (nn.Conv2d, nn.ConvTranspose2d)):
        # print(type(self.running_inc), type(self.running_outc), type(self.running_kernel_size), type(outs[2]), type(self.running_groups))
        self.running_flops = (self.running_inc * self.running_outc *
                              self.running_kernel_size * self.running_kernel_size *
                              outs[2] * outs[3] / self.running_groups)
        # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    elif isinstance(self, nn.Linear):
        self.running_flops = self.running_inc * self.running_outc
        # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    elif isinstance(self, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        # NOTE: this function is correct only when stride == kernel size
        self.running_flops = self.running_inc * ins[2] * ins[3]
        # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    return


def add_mac_hooks(m):
    global model_mac_hooks
    model_mac_hooks.append(
        m.register_forward_hook(lambda m, input, output: module_mac(
            m, input, output)))


def remove_mac_hooks():
    global model_mac_hooks
    for h in model_mac_hooks:
        h.remove()
    model_mac_hooks = []


def set_model_mode(model, mode):
    if hasattr(model, 'module'):
        model.module.set_mode(mode)
    else:
        model.set_mode(mode)


def print_gate_stats(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    for n, m in model.named_modules():
        if isinstance(m, MultiHeadGate) and getattr(m, 'print_gate', None) is not None:
            logging.info('{}: {}'.format(n, m.print_gate.sum(0)))
