'''
A modification of the ResNet module implemented in torchvision to work on 1D sequences.

Original code:
https://github.com/pytorch/vision/blob/7c077f6a986f05383bcb86b535aedb5a63dd5c4b/torchvision/models/resnet.py

'''
import torch
import torch.nn as nn

from . import model, AbstractLit, HierarchicalClassifier


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FeatureReduction(nn.Module):

    def __init__(self, inplanes, planes):
        super(FeatureReduction, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResNet(AbstractLit):

    def __init__(self, hparams):
        if not hasattr(hparams, 'zero_init_residual'):
            hparams.zero_init_residual = False
        if not hasattr(hparams, 'groups'):
            hparams.groups = 1
        if not hasattr(hparams, 'width_per_group'):
            hparams.width_per_group = 64
        if not hasattr(hparams, 'replace_stride_with_dilation'):
            hparams.replace_stride_with_dilation = None
        if not hasattr(hparams, 'norm_layer'):
            hparams.norm_layer = nn.BatchNorm1d
        if not hasattr(hparams, 'bottleneck'):
            hparams.bottleneck = True
        if not hasattr(hparams, 'simple_clf'):
            hparams.simple_clf = False
        if not hasattr(hparams, 'dropout_clf'):
            hparams.dropout_clf = False

        super(ResNet, self).__init__(hparams)

        replace_stride_with_dilation = hparams.replace_stride_with_dilation
        block = hparams.block
        layers = hparams.layers
        norm_layer = hparams.norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None or replace_stride_with_dilation is False:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        elif replace_stride_with_dilation is True:
            replace_stride_with_dilation = [True, True, True]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        input_nc = getattr(hparams, 'input_nc', None)
        n_emb = 8
        self.embedding = nn.Embedding(input_nc, n_emb)

        self.conv1 = nn.Conv1d(n_emb, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        n_output_channels = 512 * block.expansion
        if hparams.bottleneck:
            self.bottleneck = FeatureReduction(n_output_channels, 64 * block.expansion)
            n_output_channels = 64 * block.expansion
        else:
            self.bottleneck = None

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if hparams.tgt_tax_lvl == 'all':
            self.fc = HierarchicalClassifier(n_output_channels, hparams.n_taxa_all)
        elif hparams.simple_clf:
            self.fc = nn.Linear(n_output_channels, hparams.n_outputs)
        elif hparams.dropout_clf:
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(n_output_channels, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, hparams.n_outputs),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(n_output_channels, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, hparams.n_outputs),
            )


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if hparams.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def reconfigure_outputs(self, outputs_map):
        """
        Reconfigure final layer to produce more outputs.

        For example, use weights from a phylum-level classifer for initializing a class-level classifier.

        This is done by providing an array that maps from the original taxonomic level to the
        new taxonomic level


        If original output layer weights look like:

        [ [a b c],       # this corresponds to three phylum in our example
          [d e f],
          [g h i] ]

        and we reconfigure to output for 6 classes, where classes [0, 1] belong to phylum 0, [2, 3] belong to phylum 1,
        and [4, 5] belong to phylum 2 i.e. outputs_map = [0, 0, 1, 1, 2, 2], then the new output layer weights would look like:

        [ [a b c],
          [a b c],
          [d e f],
          [d e f],
          [g h i],
          [g h i] ]

        Args
            outputs_map (array)         : a mapping of original layer output to new layer output
        """
        outputs_map = torch.as_tensor(outputs_map)
        self.hparams.n_outputs = len(outputs_map)
        final_fc = self.fc
        if isinstance(self.fc, nn.Sequential):
            final_fc = self.fc[-1]
        new_fc = nn.Linear(final_fc.in_features, self.hparams.n_outputs)
        with torch.no_grad():
            for i in range(final_fc.out_features):
                mask = outputs_map == i
                new_fc.weight[mask, :] = final_fc.weight[i, :]
                new_fc.bias[mask] = final_fc.bias[i] - torch.log(mask.sum().float())
        if isinstance(self.fc, nn.Sequential):
            self.fc[-1] = new_fc
        else:
            self.fc = new_fc

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.hparams.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.hparams.groups,
                            self.hparams.width_per_group, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.hparams.groups,
                                base_width=self.hparams.width_per_group, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        


        if self.bottleneck is not None:
            x = self.bottleneck(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


@model('resnet9')
class ResNet9(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = BasicBlock
        hparams.layers = [1, 1, 1, 1]
        super().__init__(hparams)


@model('resnet18')
class ResNet18(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = BasicBlock
        hparams.layers = [2, 2, 2, 2]
        super().__init__(hparams)

@model('resnet26')
class ResNet26(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = Bottleneck
        hparams.layers = [2, 2, 2, 2]
        super().__init__(hparams)


@model('resnet34')
class ResNet34(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = BasicBlock
        hparams.layers = [3, 4, 6, 3]
        super().__init__(hparams)


@model('resnet50')
class ResNet50(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = Bottleneck
        hparams.layers = [3, 4, 6, 3]
        super().__init__(hparams)


@model('resnet74')
class ResNet74(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = Bottleneck
        hparams.layers = [3, 4, 14, 3]
        super().__init__(hparams)


@model('resnet101')
class ResNet101(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = Bottleneck
        hparams.layers = [3, 4, 23, 3]
        super().__init__(hparams)


@model('resnet152')
class ResNet152(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = Bottleneck
        hparams.layers = [3, 8, 36, 3]
        super().__init__(hparams)


@model('resnext50_32x4d')
class ResNeXt50_32x4d(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = Bottleneck
        hparams.layers = [3, 4, 6, 3]
        hparams.groups = 32
        hparams.width_per_group = 4
        super().__init__(hparams)


@model('resnext101_32x8d')
class ResNeXt101_32x8d(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = Bottleneck
        hparams.layers = [3, 4, 23, 3]
        hparams.groups = 32
        hparams.width_per_group = 8
        super().__init__(hparams)


@model('wide_resnet50_2')
class Wide_ResNet50_2(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = Bottleneck
        hparams.layers = [3, 4, 6, 3]
        hparams.width_per_group = 128
        super().__init__(hparams)


@model('wide_resnet101_2')
class Wide_ResNet101_2(ResNet):

    def __init__(self, hparams):
        hparams = self.check_hparams(hparams)
        hparams.block = Bottleneck
        hparams.layers = [3, 4, 23, 3]
        hparams.width_per_group = 128
        super().__init__(hparams)


class ResNetFeatures(nn.Module):
    """
    A class for using only the convolutional layers of a ResNet.
    This model should not be trained. See resnet_feat.py for
    trainable ResNet feature models.
    """

    _layers = ('embedding',
               'conv1',
                'bn1',
                'relu',
                'maxpool',
                'layer1',
                'layer2',
                'layer3',
                'layer4',
                'bottleneck',
                'avgpool')

    def __init__(self, resnet):
        super().__init__()
        self.hparams = resnet.hparams
        for layer in self._layers:
            setattr(self, layer, getattr(resnet, layer))

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.bottleneck is not None:
            x = self.bottleneck(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ResNetClassifier(nn.Module):
    """
    A class for using only the fully-connected classifier layer of a ResNet.
    """

    def __init__(self, resnet):
        super().__init__()
        self.fc = resnet.fc

    def forward(self, x):
        x = self.fc(x)
        return x
