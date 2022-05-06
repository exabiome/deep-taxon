'''
A modification of the VGG module implemented in torchvision to work on 1D sequences.

Original code:
https://github.com/pytorch/vision/blob/7c077f6a986f05383bcb86b535aedb5a63dd5c4b/torchvision/models/vgg.py
'''

import torch.nn as nn
import torch

from . import model, AbstractLit


class VGG(AbstractLit):
    def __init__(self, hparams):
        super().__init__(hparams)
        hparams = self.check_hparams(hparams)
        self.input_nc = getattr(hparams, 'input_nc', None)
        n_outputs = getattr(hparams, 'n_outputs', 2)
        self.features = self.make_layers(hparams.cfg, hparams.batch_norm)
        self.avgpool = nn.AdaptiveAvgPool1d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = self.input_nc
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv1d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


@model('vgg11')
class VGG11(VGG):
    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        hparams.batch_norm = False
        super().__init__(hparams)

@model('vgg11_bn')
class VGG11_bn(VGG):
    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        hparams.batch_norm = True
        super().__init__(hparams)

@model('vgg13')
class VGG13(VGG):
    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        hparams.batch_norm = False
        super().__init__(hparams)

@model('vgg13_bn')
class VGG13_bn(VGG):
    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        hparams.batch_norm = True
        super().__init__(hparams)

@model('vgg16')
class VGG16(VGG):
    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        hparams.batch_norm = False
        super().__init__(hparams)

@model('vgg16_bn')
class VGG16_bn(VGG):
    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        hparams.batch_norm = True
        super().__init__(hparams)

@model('vgg19')
class VGG19(VGG):
    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        hparams.batch_norm = False
        super().__init__(hparams)

@model('vgg19_bn')
class VGG19_bn(VGG):
    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        hparams.batch_norm = True
        super().__init__(hparams)