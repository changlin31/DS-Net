import time

import numpy as np
import torch
import torch.nn as nn

try:
    import apex
    from apex.parallel.sync_batchnorm import SyncBatchNorm
    has_apex = True
except:
    from torch.nn.modules.batchnorm import SyncBatchNorm
    has_apex = False

model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 95
params_space = 15
macs_space = 15
seconds_space = 15

num_forwards = 10


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.time = self.end - self.start
        if self.verbose:
            print('Elapsed time: %f ms.' % self.time)


def get_params(self):
    """get number of params in module"""
    return np.sum(
        [np.prod(list(w.size())) for w in self.parameters()])


def run_forward(self, input, use_cuda=True):
    with Timer() as t:
        for _ in range(num_forwards):
            self.forward(*input)
            if use_cuda:
                torch.cuda.synchronize()
    return int(t.time * 1e9 / num_forwards)


def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        'kernel_size': 'k',
        'stride': 's',
        'padding': 'pad',
        'bias': 'b',
        'groups': 'g',
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name


def module_profiling(self, input, output, verbose, use_cuda=True):
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
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input, use_cuda=use_cuda)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.ConvTranspose2d):
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input, use_cuda=use_cuda)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.Linear):
        self.n_macs = ins[1] * outs[1] * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input, use_cuda=use_cuda)
        self.name = self.__repr__()
    elif isinstance(self, nn.AvgPool2d) or isinstance(self, nn.MaxPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = run_forward(self, input, use_cuda=use_cuda)
        self.name = self.__repr__()
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = run_forward(self, input, use_cuda=use_cuda)
        self.name = self.__repr__()
    else:
        # This works only in depth-first travel of modules.
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        # num_children = 0
        # for m in self.children():
        #     if getattr(m, 'n_macs', None) is None and verbose \
        #             and type(m) not in [nn.Sequential, nn.ModuleList, nn.BatchNorm2d]:
        #         print('WARNING: leaf module {} not used in forward.'.format(type(m)))
        #     self.n_macs += getattr(m, 'n_macs', 0) if getattr(m, 'n_macs',
        #                                                       0) is not None else 0
        #     self.n_params += getattr(m, 'n_params', 0) if getattr(m, 'n_macs',
        #                                                           0) is not None else 0
        #     self.n_seconds += getattr(m, 'n_seconds', 0)
        #     num_children += 1
        #     if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
        #         mac, param, sec, child = _count_sequential(m, verbose)
        #         self.n_macs += mac
        #         self.n_params += param
        #         self.n_seconds += sec
        #         num_children += child
        ignore_zeros_t = [
            nn.BatchNorm2d, nn.Dropout2d, nn.Dropout,
            nn.Sequential, nn.ModuleList,
            nn.ReLU6, nn.ReLU, nn.MaxPool2d,
            nn.modules.padding.ZeroPad2d, nn.modules.activation.Sigmoid,
            SyncBatchNorm,
            # NonzeroBatchNorm2d, SplitMABN,
            nn.GroupNorm
        ]
        if (not getattr(self, 'ignore_model_profiling', False) and
            self.n_macs == 0 and
            t not in ignore_zeros_t) and verbose:
            print(
                'WARNING: leaf module {} has zero n_macs.'.format(type(self)))
        return
    if verbose:
        print(
            self.name.ljust(name_space, ' ') +
            '{:,}'.format(self.n_params).rjust(params_space, ' ') +
            '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
            '{:,}'.format(self.n_seconds).rjust(seconds_space, ' '))
    return


def _count_sequential(m, verbose):
    assert isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList)
    n_macs = 0
    n_params = 0
    n_seconds = 0
    num_children = 0
    for seq_m in m:
        if getattr(seq_m, 'n_macs', None) is None and verbose \
                and type(m) not in [nn.Sequential, nn.ModuleList, nn.BatchNorm2d]:
            print('WARNING: leaf module {} not used in forward.'.format(
                type(seq_m)))
        n_macs += getattr(seq_m, 'n_macs', 0) if getattr(seq_m, 'n_macs',
                                                         0) is not None else 0
        n_params += getattr(seq_m, 'n_params', 0) if getattr(seq_m, 'n_macs',
                                                             0) is not None else 0
        n_seconds += getattr(seq_m, 'n_seconds', 0)
        num_children += 1
        if isinstance(seq_m, nn.Sequential) or isinstance(seq_m, nn.ModuleList):
            mac, param, sec, child = _count_sequential(seq_m, verbose)
            n_macs += mac
            n_params += param
            n_seconds += sec
            num_children += child
    return n_macs, n_params, n_seconds, num_children


def add_profiling_hooks(m, verbose, use_cuda=True):
    global model_profiling_hooks
    model_profiling_hooks.append(
        m.register_forward_hook(lambda m, input, output: module_profiling(
            m, input, output, verbose=verbose, use_cuda=use_cuda)))


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(model, height, width, batch=1, channel=3, use_cuda=True,
                    verbose=True):
    """ Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).

    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool

    Returns:
        macs: int
        params: int

    """
    model.eval()
    data = torch.rand(batch, channel, height, width)
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.cuda() if use_cuda else model.cpu()
    data = data.cuda() if use_cuda else data.cpu()
    model.apply(lambda m: add_profiling_hooks(m, verbose=verbose, use_cuda=use_cuda))
    if verbose:
        print(
            'Item'.ljust(name_space, ' ') +
            'params'.rjust(macs_space, ' ') +
            'macs'.rjust(macs_space, ' ') +
            'nanosecs'.rjust(seconds_space, ' '))
        print(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
    model(data)
    if verbose:
        print(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
        model.n_macs = 0
        model.n_params = 0
        model.n_seconds = 0
        num_children = 0
        for n, m in model.named_modules():
            if getattr(m, 'n_macs', None) is None and verbose \
                    and type(m) not in [nn.Sequential, nn.ModuleList, nn.BatchNorm2d]:
                print('WARNING: leaf module {} not used in forward.'.format(type(m)))
            model.n_macs += getattr(m, 'n_macs', 0) if getattr(m, 'n_macs',
                                                              0) is not None else 0
            model.n_params += getattr(m, 'n_params', 0) if getattr(m, 'n_macs',
                                                                  0) is not None else 0
            model.n_seconds += getattr(m, 'n_seconds', 0)
            num_children += 1
            if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
                mac, param, sec, child = _count_sequential(m, verbose)
                model.n_macs += mac
                model.n_params += param
                model.n_seconds += sec
                num_children += child
        print(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
        print(
            'Total'.ljust(name_space, ' ') +
            '{:,}'.format(model.n_params).rjust(params_space, ' ') +
            '{:,}'.format(model.n_macs).rjust(macs_space, ' ') +
            '{:,}'.format(model.n_seconds).rjust(seconds_space, ' '))
    remove_profiling_hooks()
    return model.n_macs, model.n_params
