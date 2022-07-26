from functools import partial
from typing import Any, Callable, List, Optional, Sequence
from argparse import Namespace

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.ops.misc import ConvNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification
from torchvision.models._api import Weights, WeightsEnum

from . import model, AbstractLit


__all__ = [
    "ConvNeXtTiny",
    "ConvNeXtSmall",
    "ConvNeXtBase",
    "ConvNeXtLarge"
]


class LayerNorm1d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # x = x.permute(0, 2, 3, 1)
        x = x.permute(0, 2, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # x = x.permute(0, 3, 1, 2)
        x = x.permute(0, 2, 1)
        return x


class CNBlock(AbstractLit):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 2, 1]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        hparams: Namespace
    ) -> None:
        super().__init__()

        if getattr(hparams, 'layer_scale', None) is None:
            hparams.layer_scale = 1e-6
        if getattr(hparams, 'block', None) is None:
            hparams.block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm1d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = hparams.block_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                n_emb,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
                conv_layer=nn.Conv1d
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in hparams.block_setting)
        stage_block_id = 0
        for cnf in hparams.block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = hparams.stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, hparams.layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv1d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        input_nc = getattr(hparams, 'input_nc', None)
        n_emb = 8

        self.embedding = nn.Embedding(input_nc, n_emb)

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        lastblock = hparams.block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, hparams.n_outputs)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _block_setting(in1, in2, in3, in4, l1=3, l2=3, l3=27, l4=3):
    return [
        CNBlockConfig(in1, in2, l1),
        CNBlockConfig(in2, in3, l2),
        CNBlockConfig(in3, in4, l3),
        CNBlockConfig(in4, None, l4),
    ]


@model('convnext_tiny')
class ConvNeXtTiny(ConvNeXt):
    def __init__(self, hparams):
        hparams.block_setting = _block_setting(96, 192, 384, 768, l3=9)
        hparams.stochastic_depth_prob = 0.1
        super().__init__(hparams)


@model('convnext_small')
class ConvNeXtSmall(ConvNeXt):
    def __init__(self, hparams):
        hparams.block_setting = _block_setting(96, 192, 384, 768)
        hparams.stochastic_depth_prob = 0.4
        super().__init__(hparams)


@model('convnext_base')
class ConvNeXtBase(ConvNeXt):
    def __init__(self, hparams):
        hparams.block_setting = _block_setting(128, 256, 512, 1024)
        hparams.stochastic_depth_prob = 0.5
        super().__init__(hparams)


@model('convnext_large')
class ConvNeXtLarge(ConvNeXt):
    def __init__(self, hparams):
        hparams.block_setting = _block_setting(192, 384, 768, 1536)
        hparams.stochastic_depth_prob = 0.5
        super().__init__(hparams)
