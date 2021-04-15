import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyn_slim.models.dyn_slim_blocks import DSInvertedResidual, DSDepthwiseSeparable, set_exist_attr, MultiHeadGate
from dyn_slim.models.dyn_slim_ops import DSConv2d, DSpwConv2d, DSBatchNorm2d, DSLinear, DSAdaptiveAvgPool2d
from dyn_slim.models.dyn_slim_stages import DSStage
from timm.models.layers import Swish
from timm.models.registry import register_model

__all__ = ['DSNet']

from dyn_slim.utils import efficientnet_init_weights


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


choices_cfgs = {  # outc, layer, kernel, stride, type, has_gate
    'slimmable_mbnet_v1_bn_uniform': [
        [[16, 24], 1, 3, 2],
        [[32, 88], 1, 3, 1, 'ds', False],
        [[48, 168], 2, 3, 2, 'ds', False],
        [[96, 264], 2, 3, 2, 'ds', False],
        [list(range(224, 640 + 1, 32)), 6, 3, 2, 'ds', True],
        [list(range(736, 1152 + 1, 32)), 2, 3, 2, 'ds', False],
        [],  # no head
    ],
}


class DSNet(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3,
                 choices_cfg=None, act_layer=nn.ReLU, noskip=False, drop_rate=0.,
                 drop_path_rate=0., se_ratio=0.25, norm_layer=DSBatchNorm2d,
                 norm_kwargs=None, bias=False, has_head=True, **kwargs):
        super(DSNet, self).__init__()
        assert drop_path_rate == 0., 'drop connect not supported yet!'
        # logging.warning('Following args are not used when building DSNet:', kwargs)
        norm_kwargs = norm_kwargs or {}
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self._in_chs_list = [in_chans]

        # Stem
        stem_size, _, kernel_size, stride = choices_cfg[0]
        self.conv_stem = DSConv2d(self._in_chs_list,
                                  stem_size,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  bias=bias)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs_list = stem_size

        # Middle stages (IR/ER/DS Blocks)
        self.blocks = nn.ModuleList()
        for (out_channels_list, layer_num, kernel_size, stride, stage_type, has_gate) in choices_cfg[1:-1]:
            self.blocks.append(DSStage(stage_type=stage_type,
                                       in_channels_list=self._in_chs_list,
                                       out_channels_list=out_channels_list,
                                       kernel_size=kernel_size,
                                       layer_num=layer_num,
                                       stride=stride,
                                       act_layer=act_layer,
                                       noskip=noskip,
                                       se_ratio=se_ratio,
                                       norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs,
                                       drop_path_rate=drop_path_rate,
                                       bias=bias,
                                       has_gate=has_gate))
            self._in_chs_list = out_channels_list

        # Head + Pooling
        if has_head and len(choices_cfg[-1]) > 0:  # no head in mbnetv1
            self.has_head = True
            self.num_features = [choices_cfg[-1][0][-1]]
            self.conv_head = DSpwConv2d(self._in_chs_list, self.num_features, bias=bias)
            self.bn2 = norm_layer(self.num_features, **norm_kwargs)
            self.act2 = act_layer(inplace=True)
        else:
            self.has_head = False
            self.num_features = self._in_chs_list
        self.global_pool = DSAdaptiveAvgPool2d(1, channel_list=self.num_features)

        # Classifier
        self.classifier = DSLinear(self.num_features, self.num_classes)

        with torch.no_grad():
            efficientnet_init_weights(self)
            self.init_residual_norm()
        self.set_mode('largest')
        self.head_channel_choice = None
        self.stem_channel_choice = None

    def set_mode(self, mode, seed=None, choice=None):
        self.mode = mode
        if seed is not None:
            random.seed(seed)
            seed += 1
        assert mode in ['largest', 'smallest', 'dynamic', 'uniform']
        for m in self.modules():
            set_exist_attr(m, 'mode', mode)
        if mode == 'largest':
            self.channel_choice = -1
            if self.has_head:
                self.set_module_choice(self.conv_head)
        elif mode == 'smallest' or mode == 'dynamic':
            self.channel_choice = 0
            if self.has_head:
                self.set_module_choice(self.conv_head)
        elif mode == 'uniform':
            self.channel_choice = 0
            if self.has_head:
                self.set_module_choice(self.conv_head)
            self.channel_choice = 0
            if choice is not None:
                self.random_choice = choice
            else:
                self.random_choice = random.randint(1, 13)

        self.set_module_choice(self.conv_stem)
        self.set_module_choice(self.bn1)

    def set_module_choice(self, m):
        set_exist_attr(m, 'channel_choice', self.channel_choice)

    def set_self_choice(self, m):
        self.channel_choice = m.channel_choice

    def init_residual_norm(self):
        for n, m in self.named_modules():
            if isinstance(m, DSInvertedResidual) or isinstance(m, DSDepthwiseSeparable):
                if m.has_residual:
                    logging.info('set block {} bn weight to zero'.format(n))
                m.init_residual_norm(level='block')

    def get_gate(self):
        gate = nn.ModuleList()
        for n, m in self.named_modules():
            if isinstance(m, MultiHeadGate) and m.has_gate:
                gate += [m.gate]
        return gate

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.global_pool = DSAdaptiveAvgPool2d(1, channel_list=self.num_features)
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(),
            num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for idx, stage in enumerate(self.blocks):  # TODO: Optimize code
            if idx >= 3 and self.mode == 'uniform':  # 3 for mbnet, 4 for effnet
                setattr(stage.first_block, 'random_choice', self.random_choice)
            else:
                setattr(stage.first_block, 'random_choice', 0)
            self.set_module_choice(stage)
            if idx >= 4 and self.mode == 'uniform':  # 4 for mbnet, 5 for effnet
                setattr(stage, 'channel_choice', self.random_choice)
            x = stage(x)
            self.set_self_choice(stage)
        if self.has_head:
            self.set_module_choice(self.conv_head)
            self.set_module_choice(self.bn2)
            x = self.conv_head(x)
            x = self.bn2(x)
            x = self.act2(x)
        self.set_module_choice(self.global_pool)
        self.set_module_choice(self.classifier)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        if self.drop_rate > 0. and self.mode == 'largest':
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x


# --------------------------------
@register_model
def slimmable_mbnet_v1_bn_uniform(pretrained=False, **kwargs):
    kwargs['noskip'] = True
    model = DSNet(
        choices_cfg=choices_cfgs['slimmable_mbnet_v1_bn_uniform'],
        pretrained=pretrained,
        act_layer=Swish,
        norm_layer=DSBatchNorm2d,
        **kwargs)
    return model
