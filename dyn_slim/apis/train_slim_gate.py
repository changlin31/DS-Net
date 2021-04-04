import time

from dyn_slim.models.dyn_slim_blocks import MultiHeadGate
from dyn_slim.utils.slim_net_profiling import add_flops

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP

    has_apex = False
from timm.utils import *

import torch
import torch.nn as nn
import torchvision.utils

model_mac_hooks = []


def train_epoch_slim_gate(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False, model_ema=None,
        optimizer_step=1):
    pass


def reduce_list_tensor(tensor_l, world_size):
    ret_l = []
    for tensor in tensor_l:
        ret_l.append(reduce_tensor(tensor, world_size))
    return ret_l


def validate_gate(model, loader, loss_fn, args, log_suffix=''):
    pass


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
    # NOTE: There are some difference between type and isinstance, thus please
    # be careful.
    t = type(self)
    if isinstance(self, nn.Conv2d):
        # print(type(self.running_inc), type(self.running_outc), type(self.running_kernel_size), type(outs[2]), type(self.running_groups))
        self.running_flops = (self.running_inc * self.running_outc *
                              self.running_kernel_size * self.running_kernel_size *
                              outs[2] * outs[3] // self.running_groups)
    elif isinstance(self, nn.ConvTranspose2d):
        self.running_flops = (self.running_inc * self.running_outc *
                              self.running_kernel_size * self.running_kernel_size *
                              outs[2] * outs[3] // self.running_groups)
    elif isinstance(self, nn.Linear):
        self.running_flops = self.running_inc * self.running_outc
    elif isinstance(self, nn.AvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.running_flops = ins[1] * ins[2] * ins[3]
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.running_flops = ins[1] * ins[2] * ins[3]
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
        if isinstance(m, MultiHeadGate):
            if hasattr(m, 'keep_gate'):
                # print('=' * 10)
                logging.info('{}: {}'.format(n, m.print_gate.sum(0)))
