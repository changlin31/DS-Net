import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyn_slim.models.dyn_slim_ops import DSpwConv2d, DSdwConv2d, DSBatchNorm2d
from timm.models.layers import sigmoid


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DSInvertedResidual(nn.Module):

    def __init__(self, in_channels_list, out_channels_list, kernel_size, layer_num, stride=1, dilation=1,
                 act_layer=nn.ReLU, noskip=False, exp_ratio=6.0, se_ratio=0.25,
                 norm_layer=DSBatchNorm2d, norm_kwargs=None, conv_kwargs=None,
                 drop_path_rate=0., bias=False, has_gate=False):
        super(DSInvertedResidual, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.layer_num = layer_num
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_channels_list = [make_divisible(inc * exp_ratio) for inc in in_channels_list]
        self.has_residual = not noskip
        self.drop_path_rate = drop_path_rate
        self.has_gate = has_gate
        if self.has_residual:
            if in_channels_list[-1] != out_channels_list[-1] or stride == 2:
                downsample_layers = []
                if stride == 2:
                    downsample_layers += [nn.AvgPool2d(2, 2, ceil_mode=True,
                                                       count_include_pad=False)]
                if in_channels_list[-1] != out_channels_list[-1]:
                    downsample_layers += [DSpwConv2d(in_channels_list,
                                                     out_channels_list,
                                                     bias=bias)]
                self.downsample = nn.Sequential(*downsample_layers)
            else:
                self.downsample = None

        # Point-wise expansion
        self.conv_pw = DSpwConv2d(in_channels_list, mid_channels_list, bias=bias)
        self.bn1 = norm_layer(mid_channels_list, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = DSdwConv2d(mid_channels_list,
                                  stride=stride,
                                  dilation=dilation,
                                  bias=bias)
        self.bn2 = norm_layer(mid_channels_list, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Channel attention and gating
        self.gate = MultiHeadGate(mid_channels_list,
                                se_ratio=se_ratio,
                                channel_gate_num=len(out_channels_list) if has_gate else 0)

        # Point-wise linear projection
        self.conv_pwl = DSpwConv2d(mid_channels_list, out_channels_list, bias=bias)
        self.bn3 = norm_layer(out_channels_list, **norm_kwargs)
        self.channel_choice = -1
        self.mode = 'largest'
        self.next_channel_choice = None
        self.last_feature = None
        self.random_choice = 0
        self.init_residual_norm()

    def init_residual_norm(self, level='block'):
        if self.has_residual:
            if level == 'block':
                self.bn3.set_zero_weight()
            elif level == 'channel':
                self.bn1.set_zero_weight()
                self.bn3.set_zero_weight()

    def feature_module(self, location):
        if location == 'post_exp':
            return 'act1'
        return 'conv_pwl'

    def feature_channels(self, location):
        if location == 'post_exp':
            return self.conv_pw.out_channels
        # location == 'pre_pw'
        return self.conv_pwl.in_channels

    def get_last_stage_distill_feature(self):
        return self.last_feature

    def forward(self, x):
        self._set_gate()

        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Channel attention and gating
        x = self.gate(x)
        if self.has_gate:
            self.prev_channel_choice = self.channel_choice
            self.channel_choice = self._new_gate()
            self._set_gate(set_pwl=True)
        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            if self.drop_path_rate > 0. and self.mode == 'largest':
                # Only apply drop_path on largest model
                x = drop_path(x, self.training, self.drop_path_rate)
            x += residual

        return x

    def _set_gate(self, set_pwl=False):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', self.channel_choice)
        if set_pwl:
            self.conv_pwl.prev_channel_choice = self.prev_channel_choice
            if self.downsample is not None:
                for n, m in self.downsample.named_modules():
                    set_exist_attr(m, 'prev_channel_choice', self.prev_channel_choice)

    def _new_gate(self):
        if self.mode == 'largest':
            return -1
        elif self.mode == 'smallest':
            return 0
        elif self.mode == 'uniform':
            return self.random_choice
        elif self.mode == 'random':
            return random.randint(0, len(self.out_channels_list) - 1)
        elif self.mode == 'dynamic':
            return self.se.get_gate()

    def get_gate(self):
        return self.channel_choice


class DSDepthwiseSeparable(nn.Module):

    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, layer_num, stride=1, dilation=1,
                 act_layer=nn.ReLU, noskip=False, se_ratio=0.25,
                 norm_layer=DSBatchNorm2d, norm_kwargs=None, conv_kwargs=None,
                 drop_path_rate=0., bias=False, has_gate=False):
        super(DSDepthwiseSeparable, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.layer_num = layer_num
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        self.has_residual = not noskip
        self.drop_path_rate = drop_path_rate
        self.has_gate = has_gate
        if self.has_residual:
            if in_channels_list[-1] != out_channels_list[-1] or stride == 2:
                downsample_layers = []
                if stride == 2:
                    downsample_layers += [nn.AvgPool2d(2, 2, ceil_mode=True,
                                                       count_include_pad=False)]
                if in_channels_list[-1] != out_channels_list[-1]:
                    downsample_layers += [DSpwConv2d(in_channels_list,
                                                     out_channels_list,
                                                     bias=bias)]
                self.downsample = nn.Sequential(*downsample_layers)
            else:
                self.downsample = None
        # Depth-wise convolution
        self.conv_dw = DSdwConv2d(in_channels_list,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  dilation=dilation,
                                  bias=bias)
        self.bn1 = norm_layer(in_channels_list, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Channel attention and gating
        self.gate = MultiHeadGate(in_channels_list,
                                se_ratio=se_ratio,
                                channel_gate_num=len(out_channels_list) if has_gate else 0)

        # Point-wise convolution
        self.conv_pw = DSpwConv2d(in_channels_list, out_channels_list, bias=bias)
        self.bn2 = norm_layer(out_channels_list, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.channel_choice = -1
        self.mode = 'largest'
        self.next_channel_choice = None
        self.random_choice = 0
        self.init_residual_norm()

    def init_residual_norm(self, level='block'):
        if self.has_residual:
            if level == 'block':
                self.bn2.set_zero_weight()
            elif level == 'channel':
                self.bn1.set_zero_weight()
                self.bn2.set_zero_weight()

    def forward(self, x):
        self._set_gate()
        residual = x

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Channel attention and gating
        x = self.gate(x)
        if self.has_gate:
            self.channel_choice = self._new_gate()
            self._set_gate()
        # Point-wise convolution
        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            if self.drop_path_rate > 0. and self.mode == 'largest':
                # Only apply drop_path on largest model
                x = drop_path(x, self.training, self.drop_path_rate)
            x += residual

        return x

    def _set_gate(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', self.channel_choice)

    def _new_gate(self):
        if self.mode == 'largest':
            return -1
        elif self.mode == 'smallest':
            return 0
        elif self.mode == 'uniform':
            return self.random_choice
        elif self.mode == 'random':
            return random.randint(0, len(self.out_channels_list) - 1)
        elif self.mode == 'dynamic':
            return self.se.get_gate()

    def get_gate(self):
        return self.channel_choice


class MultiHeadGate(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, attn_act_fn=sigmoid, divisor=1, channel_gate_num=None):
        super(MultiHeadGate, self).__init__()
        self.attn_act_fn = attn_act_fn
        self.channel_gate_num = channel_gate_num
        reduced_chs = make_divisible((reduced_base_chs or in_chs[-1]) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = DSpwConv2d(in_chs, [reduced_chs], bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = DSpwConv2d([reduced_chs], in_chs, bias=True)

        #  Dynamic Gate and Training Stage II will be released soon

        self.mode = 'largest'
        self.channel_choice = None
        self.initialized = False
        if self.attn_act_fn == 'tanh':
            nn.init.zeros_(self.conv_expand.weight)
            nn.init.zeros_(self.conv_expand.bias)

    def forward(self, x):
        x_pool = self.avg_pool(x)
        x_reduced = self.conv_reduce(x_pool)
        x_reduced = self.act1(x_reduced)
        attn = self.conv_expand(x_reduced)
        if self.attn_act_fn == 'tanh':
            attn = (1 + attn.tanh())
        else:
            attn = self.attn_act_fn(attn)
        x = x * attn

        return x

    def get_gate(self):
        return self.channel_choice


def gumbel_softmax(logits, tau=1, hard=False, dim=-1, training=True):
    """ See `torch.nn.functional.gumbel_softmax()` """
    if training:
        noise = torch.rand_like(logits)
        gumbels = -(-noise.log()).log()
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    else:
        gumbels = logits
    y_soft = gumbels.softmax(-1)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        # y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
        return ret, index
    else:
        # Reparametrization trick.
        return y_soft


def drop_path(inputs, training=False, drop_path_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_path_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)
