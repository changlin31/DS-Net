import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['DSConv2d', 'DSdwConv2d', 'DSpwConv2d', 'DSBatchNorm2d', 'DSLinear', 'DSAvgPool2d', 'DSAdaptiveAvgPool2d']


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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
        if not isinstance(in_channels_list, (list, tuple)):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, (list, tuple)):
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
        assert self.groups in (1, self.out_channels), \
            'only support regular conv, pwconv and dwconv'
        padding = ((self.stride[0] - 1) + self.dilation[0] * (
                self.kernel_size[0] - 1)) // 2
        self.padding = (padding, padding)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.channel_choice = -1  # dynamic channel list index
        self.in_chn_static = len(in_channels_list) == 1
        self.out_chn_static = len(out_channels_list) == 1
        self.running_inc = self.in_channels if self.in_chn_static else None
        self.running_outc = self.out_channels if self.out_chn_static else None
        self.running_kernel_size = self.kernel_size[0]
        self.running_groups = self.groups
        self.mode = 'largest'
        self.prev_channel_choice = None
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()
        self.out_channels_list_tensor = torch.from_numpy(
            np.array(self.out_channels_list)).float().cuda()

    def forward(self, x):
        if self.prev_channel_choice is None:
            self.prev_channel_choice = self.channel_choice
        if self.mode == 'dynamic' and isinstance(self.channel_choice, tuple):
            weight = self.weight
            if not self.in_chn_static:
                if isinstance(self.prev_channel_choice, int):
                    self.running_inc = self.in_channels_list[self.prev_channel_choice]
                    weight = self.weight[:, :self.running_inc]
                else:
                    self.running_inc = torch.matmul(self.prev_channel_choice[0], self.in_channels_list_tensor)
            if not self.out_chn_static:
                self.running_outc = torch.matmul(self.channel_choice[0], self.out_channels_list_tensor)

            output = F.conv2d(x,
                              weight,
                              self.bias,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups)
            if not self.out_chn_static:
                output = apply_differentiable_gate_channel(output,
                                                           self.channel_choice[0],
                                                           self.out_channels_list)
            self.prev_channel_choice = None
            self.channel_choice = -1
            return output
        else:
            if not self.in_chn_static:
                self.running_inc = x.size(1)
            if not self.out_chn_static:
                self.running_outc = self.out_channels_list[self.channel_choice]
            weight = self.weight[:self.running_outc, :self.running_inc]
            bias = self.bias[:self.running_outc] if self.bias is not None else None
            self.running_groups = 1 if self.groups == 1 else self.running_outc
            self.prev_channel_choice = None
            self.channel_choice = -1
            return F.conv2d(x,
                            weight,
                            bias,
                            self.stride,
                            self.padding,
                            self.dilation,
                            self.running_groups)


class DSpwConv2d(DSConv2d):
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 bias=True):
        super(DSpwConv2d, self).__init__(
            in_channels_list=in_channels_list,
            out_channels_list=out_channels_list,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode='zeros')


class DSdwConv2d(DSConv2d):
    def __init__(self,
                 channels_list,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=True):
        super(DSdwConv2d, self).__init__(
            in_channels_list=channels_list,
            out_channels_list=channels_list,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=channels_list[-1],
            bias=bias,
            padding_mode='zeros')


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
        self.running_inc = x.size(1)

        if self.mode == 'dynamic' and isinstance(self.channel_choice, tuple):
            self.channel_choice, idx = self.channel_choice
            running_mean = torch.zeros_like(self.aux_bn[-1].running_mean).repeat(len(self.out_channels_list), 1)
            running_var = torch.zeros_like(self.aux_bn[-1].running_var).repeat(len(self.out_channels_list), 1)
            for i in range(len(self.out_channels_list)):
                running_mean[i, :self.out_channels_list[i]] += self.aux_bn[i].running_mean
                running_var[i, :self.out_channels_list[i]] += self.aux_bn[i].running_var
            running_mean = torch.matmul(self.channel_choice, running_mean)[..., None, None].expand_as(x)
            running_var = torch.matmul(self.channel_choice, running_var)[..., None, None].expand_as(x)
            weight = self.aux_bn[-1].weight[:self.running_inc] if self.affine else None
            bias = self.aux_bn[-1].bias[:self.running_inc] if self.affine else None

            x = (x - running_mean) / torch.sqrt(running_var + self.aux_bn[-1].eps)
            x = x * weight[..., None, None].expand_as(x) + bias[..., None, None].expand_as(x)
            return apply_differentiable_gate_channel(x, self.channel_choice, self.out_channels_list)

        else:
            idx = self.out_channels_list.index(self.running_inc)
            running_mean = self.aux_bn[idx].running_mean
            running_var = self.aux_bn[idx].running_var
            weight = self.aux_bn[-1].weight[:self.running_inc] if self.affine else None
            bias = self.aux_bn[-1].bias[:self.running_inc] if self.affine else None
            return F.batch_norm(x,
                                running_mean,
                                running_var,
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
        self.in_channels_list = in_features_list
        self.out_channels_list = [out_features]
        self.channel_choice = -1
        self.mode = 'largest'
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()

    def forward(self, x):
        if self.mode == 'dynamic':
            if isinstance(self.channel_choice, tuple):
                self.channel_choice = self.channel_choice[0]
                self.running_inc = torch.matmul(self.channel_choice, self.in_channels_list_tensor)
            else:
                self.running_inc = self.in_channels_list[self.channel_choice]
            self.running_outc = self.out_features
            return F.linear(x, self.weight, self.bias)
        else:
            self.running_inc = x.size(1)
            self.running_outc = self.out_features
            weight = self.weight[:, :self.running_inc]
            return F.linear(x, weight, self.bias)


class DSAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size, channel_list):
        super(DSAdaptiveAvgPool2d, self).__init__(output_size=output_size)
        self.in_channels_list = channel_list
        self.channel_choice = -1
        self.mode = 'largest'
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()

    def forward(self, x):
        if self.mode == 'dynamic':
            if isinstance(self.channel_choice, tuple):
                self.channel_choice = self.channel_choice[0]
                self.running_inc = torch.matmul(self.channel_choice, self.in_channels_list_tensor)
            else:
                self.running_inc = self.in_channels_list[self.channel_choice]
        else:
            self.running_inc = x.size(1)
        return super(DSAdaptiveAvgPool2d, self).forward(input=x)


class DSAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride, channel_list, padding=0, ceil_mode=True, count_include_pad=False):
        super(DSAvgPool2d, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding,
                                          ceil_mode=ceil_mode, count_include_pad=count_include_pad)
        self.in_channels_list = channel_list
        self.channel_choice = -1
        self.mode = 'largest'
        self.prev_channel_choice = None
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()

    def forward(self, x):
        if self.mode == 'dynamic':
            if self.prev_channel_choice is None:
                self.prev_channel_choice = self.channel_choice
            if isinstance(self.prev_channel_choice, tuple):
                self.prev_channel_choice = self.prev_channel_choice[0]
                self.running_inc = torch.matmul(self.prev_channel_choice, self.in_channels_list_tensor)
            else:
                self.running_inc = self.in_channels_list[self.prev_channel_choice]
        else:
            self.running_inc = x.size(1)
        return super(DSAvgPool2d, self).forward(input=x)


def apply_differentiable_gate_channel(x, channel_gate, channel_list):
    ret = torch.zeros_like(x)
    if not isinstance(channel_gate, torch.Tensor):
        ret[:, :channel_list[channel_gate]] += x[:, :channel_list[channel_gate]]
    else:
        # print(channel_gate)
        for idx in range(len(channel_list)):
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            ret[:, :channel_list[idx]] += x[:, :channel_list[idx]] * (
                channel_gate[:, idx, None, None, None].expand_as(x[:, :channel_list[idx]]))
    return ret
