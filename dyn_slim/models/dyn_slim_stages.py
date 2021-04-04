import torch.nn as nn

from dyn_slim.models.dyn_slim_blocks import DSDepthwiseSeparable, DSInvertedResidual, set_exist_attr


class DSStage(nn.Module):
    def __init__(self, stage_type, in_channels_list, out_channels_list,
                 kernel_size, layer_num, stride=1, dilation=1,
                 act_layer=nn.ReLU, noskip=False, exp_ratio=6.0, se_ratio=0.25,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, conv_kwargs=None,
                 drop_path_rate=0., bias=False, has_gate=False):
        super(DSStage, self).__init__()
        self.stage_type = stage_type
        if stage_type.lower() == 'ir':
            self.first_block = DSInvertedResidual(
                in_channels_list, out_channels_list, kernel_size,
                layer_num, stride, dilation, act_layer, noskip,
                exp_ratio, se_ratio, norm_layer, norm_kwargs,
                conv_kwargs, drop_path_rate, bias=bias, has_gate=has_gate)
            residual_blocks = []
            for _ in range(layer_num - 1):
                residual_blocks.append(DSInvertedResidual(
                    out_channels_list, out_channels_list, kernel_size,
                    layer_num, 1, dilation, act_layer, noskip,
                    exp_ratio, se_ratio, norm_layer, norm_kwargs,
                    conv_kwargs, drop_path_rate, bias=bias, has_gate=False))
        else:  # stage_type.lower() == 'ds'
            self.first_block = DSDepthwiseSeparable(
                in_channels_list, out_channels_list, kernel_size,
                layer_num, stride, dilation,
                act_layer, noskip, se_ratio,
                norm_layer, norm_kwargs, conv_kwargs,
                drop_path_rate, bias=bias, has_gate=has_gate)
            residual_blocks = []
            for _ in range(layer_num - 1):
                residual_blocks.append(DSDepthwiseSeparable(
                    out_channels_list, out_channels_list, kernel_size,
                    layer_num, 1, dilation, act_layer, noskip,
                    se_ratio, norm_layer, norm_kwargs,
                    conv_kwargs, drop_path_rate, bias=bias, has_gate=False))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        self.layer_num = layer_num
        self.channel_choice = -1
        self.mode = 'largest'

    def forward(self, x):
        self.set_gate()
        x = self.first_block(x)
        self.channel_choice = self.first_block.get_gate()
        self.set_gate()
        if self.mode == 'dynamic':
            x = self.residual_blocks(x)
        else:
            for idx in range(self.layer_num - 1):
                x = self.residual_blocks[idx](x)

        return x

    def set_gate(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', self.channel_choice)
