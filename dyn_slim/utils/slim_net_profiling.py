from .model_profiling import model_profiling
from timm.models import create_model

import torch.nn as nn


def main():
    model = create_model(
            'resnet50',
            pretrained=False,
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.,
            global_pool='avg',
            bn_tf=False,
            bn_momentum=None,
            bn_eps=None,
            checkpoint_path='',
            use_se=True)

    model_profiling(model, 224, 224, use_cuda=False)

    flops = add_flops(model)

    print('model.flops', flops)


def add_flops(model):
    flops = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d) \
                or isinstance(m, nn.Linear) \
                or isinstance(m, nn.AvgPool2d) \
                or isinstance(m, nn.AdaptiveAvgPool2d):
            flops += getattr(m, 'running_flops', 0)

    return flops


if __name__ == '__main__':
    main()
