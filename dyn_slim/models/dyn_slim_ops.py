import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['DSConv2d', 'DSdwConv2d', 'DSpwConv2d',
           'DSBatchNorm2d', 'DSLinear']


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DSpwConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 bias=True):
        super(DSpwConv2d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            kernel_size=1,
            groups=1,
            bias=bias,
            padding_mode='zeros')
        padding = ((self.stride[0] - 1) + self.dilation[0] * (
                self.kernel_size[0] - 1)) // 2
        self.padding = (padding, padding)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.channel_choice = -1  # dynamic channel list index
        self.in_chn_static = len(in_channels_list) == 1
        self.out_chn_static = len(out_channels_list) == 1
        self.running_kernel_size = 1
        self.running_groups = 1
        self.mode = 'largest'
        self.prev_channel_choice = None

        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()
        self.out_channels_list_tensor = torch.from_numpy(
            np.array(self.out_channels_list)).float().cuda()

    def forward(self, x):
        # print(x.size(1))
        if self.prev_channel_choice is None:
            self.prev_channel_choice = self.channel_choice
        if self.mode == 'dynamic':
            raise NotImplementedError('Stage II not supported yet!')
        else:  # super net training mode
            self.running_inc = self.in_channels if self.in_chn_static else x.size(1)
            self.running_outc = self.out_channels if self.out_chn_static \
                else self.out_channels_list[self.channel_choice]
            weight = self.weight[:self.running_outc, :self.running_inc]
            bias = self.bias[:self.running_outc] if self.bias is not None else None
            self.prev_channel_choice = None
            self.channel_choice = -1
            return F.conv2d(x,
                            weight,
                            bias,
                            self.stride,
                            self.padding,
                            self.dilation,
                            self.groups)


class DSdwConv2d(nn.Conv2d):
    def __init__(self,
                 channels_list,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=True):
        super(DSdwConv2d, self).__init__(
            in_channels=channels_list[-1],
            out_channels=channels_list[-1],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=channels_list[-1],
            bias=bias,
            padding_mode='zeros')
        padding = ((self.stride[0] - 1) + self.dilation[0] * (
                self.kernel_size[0] - 1)) // 2
        self.padding = (padding, padding)
        self.channels_list = channels_list
        self.channel_choice = -1  # dynamic channel list index
        self.chn_static = len(channels_list) == 1
        self.running_kernel_size = self.kernel_size[0]
        self.running_groups = self.groups
        self.mode = 'largest'
        self.prev_channel_choice = None
        self.channels_list_tensor = torch.from_numpy(
            np.array(self.channels_list)).float().cuda()

    def forward(self, x):
        # print(x.size(1))
        if self.prev_channel_choice is None:
            self.prev_channel_choice = self.channel_choice
        if self.mode == 'dynamic':
            raise NotImplementedError('Stage II not supported yet!')
        else:
            channels = self.channels_list[self.channel_choice] if not self.chn_static \
                else self.channels_list[-1]
            self.running_inc = channels
            self.running_outc = channels
            self.running_groups = channels
            weight = self.weight[:channels, :]
            padding = self.padding

            bias = self.bias[:channels] if self.bias is not None else None

            groups = channels
            self.channel_choice = -1
            return F.conv2d(x,
                            weight,
                            bias,
                            self.stride,
                            padding,
                            self.dilation,
                            groups)


class DSConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        if not isinstance(in_channels_list, list):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, list):
            out_channels_list = [out_channels_list]
        super(DSConv2d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)
        padding = ((self.stride[0] - 1) + self.dilation[0] * (
                self.kernel_size[0] - 1)) // 2
        self.padding = (padding, padding)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.channel_choice = -1  # dynamic channel list index
        self.in_chn_static = len(in_channels_list) == 1
        self.out_chn_static = len(out_channels_list) == 1
        self.running_kernel_size = self.kernel_size[0]
        self.running_groups = self.groups
        self.mode = 'largest'
        self.prev_channel_choice = None
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()
        self.out_channels_list_tensor = torch.from_numpy(
            np.array(self.out_channels_list)).float().cuda()

    def forward(self, x):
        assert self.groups == 1, \
            'only support regular conv, pwconv and dwconv'
        if self.prev_channel_choice is None:
            self.prev_channel_choice = self.channel_choice
        if self.mode == 'dynamic':
            raise NotImplementedError('Stage II not supported yet!')
        else:
            self.running_inc = self.in_channels if self.in_chn_static else x.size(1)
            self.running_outc = self.out_channels if self.out_chn_static else \
                self.out_channels_list[self.channel_choice]

            weight = self.weight[:self.running_outc, :self.running_inc]
            padding = self.padding

            bias = self.bias[:self.running_outc] if self.bias is not None else None

            groups = 1 if self.groups == 1 else self.running_outc
            self.running_groups = groups
            self.channel_choice = -1
            return F.conv2d(x,
                            weight,
                            bias,
                            self.stride,
                            padding,
                            self.dilation,
                            groups)


class DSBatchNorm2d(nn.Module):
    def __init__(self,
                 num_features_list,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(DSBatchNorm2d, self).__init__()
        self.out_channels_list = num_features_list
        self.aux_bn = nn.ModuleList([
            nn.BatchNorm2d(channel, affine=False) for channel in
            self.out_channels_list[:-1]])
        self.aux_bn.append(nn.BatchNorm2d(self.out_channels_list[-1],
                                          eps=eps,
                                          momentum=momentum,
                                          affine=affine))
        self.affine = affine
        self.channel_choice = -1
        self.mode = 'largest'

    def set_zero_weight(self):
        if self.affine:
            nn.init.zeros_(self.aux_bn[-1].weight)

    def forward(self, x):
        if self.mode == 'dynamic':
            raise NotImplementedError('Stage II not supported yet!')

        self.running_inc = x.size(1)

        weight = self.aux_bn[-1].weight[:self.running_inc] if self.affine else None
        bias = self.aux_bn[-1].bias[:self.running_inc] if self.affine else None

        idx = self.out_channels_list.index(self.running_inc)

        return F.batch_norm(
            x,
            self.aux_bn[idx].running_mean,
            self.aux_bn[idx].running_var,
            weight,
            bias,
            self.training,
            self.aux_bn[-1].momentum,
            self.aux_bn[-1].eps)


class DSLinear(nn.Linear):
    def __init__(self,
                 in_features_list,
                 out_features,
                 bias=True):
        super(DSLinear, self).__init__(
            in_features=in_features_list[-1],
            out_features=out_features,
            bias=bias)
        self.out_channels_list = in_features_list
        self.channel_choice = -1
        self.mode = 'largest'
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.out_channels_list)).float().cuda()

    def forward(self, x):
        if self.mode == 'dynamic':
            raise NotImplementedError('Stage II not supported yet!')
        self.running_inc = x.size(1)
        self.running_outc = self.out_features
        weight = self.weight[:, :self.running_inc]
        return F.linear(x, weight, self.bias)
