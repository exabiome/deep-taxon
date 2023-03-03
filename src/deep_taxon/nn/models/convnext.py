from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from argparse import Namespace

import torch
from torch import nn, Tensor
from torch.nn import functional as F

try:
    from torchvision.ops.misc import ConvNormActivation, Permute
except:
    from torchvision.ops.misc import ConvNormActivation
    class Permute(nn.Module):
        def __init__(self, dims: List[int]):
            super().__init__()
            self.dims = dims

        def forward(self, x):
            return torch.permute(x, self.dims)

from torchvision.ops.stochastic_depth import StochasticDepth

from . import model, AbstractLit


__all__ = [
    "ConvNeXtTiny",
    "ConvNeXtSmall",
    "ConvNeXtBase",
    "ConvNeXtLarge"
]


class Conv1dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv1d,
        )



class LayerNorm1d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # x = x.permute(0, 2, 3, 1)
        x = x.permute(0, 2, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # x = x.permute(0, 3, 1, 2)
        x = x.permute(0, 2, 1)
        return x


class CNBlock(nn.Module):
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
        self.layer_scale = nn.Parameter(torch.ones(dim, 1) * layer_scale)
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


class ConvNeXt(AbstractLit):
    def __init__(
        self,
        hparams: Namespace,
        **kwargs
    ) -> None:
        super().__init__(hparams, **kwargs)

        stochastic_depth_prob = getattr(hparams, 'stochastic_depth_prob ', 0.0)
        layer_scale = getattr(hparams, 'layer_scale', 1e-6)
        block = getattr(hparams, 'block', CNBlock)
        norm_layer = getattr(hparams, 'norm_layer ', partial(LayerNorm1d, eps=1e-6))
        n_emb = getattr(hparams, 'n_emb ', 8)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = hparams.block_setting[0].input_channels
        layers.append(
            Conv1dNormActivation(
                n_emb,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in hparams.block_setting)
        stage_block_id = 0
        for cnf in hparams.block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
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
        x = self.embedding(x.int())
        x = x.permute(0, 2, 1)
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
    def __init__(self, hparams, **kwargs):
        hparams.block_setting = _block_setting(96, 192, 384, 768, l3=9)
        hparams.stochastic_depth_prob = 0.1
        super().__init__(hparams, **kwargs)


@model('convnext_small')
class ConvNeXtSmall(ConvNeXt):
    def __init__(self, hparams, **kwargs):
        hparams.block_setting = _block_setting(96, 192, 384, 768)
        hparams.stochastic_depth_prob = 0.4
        super().__init__(hparams, **kwargs)


@model('convnext_base')
class ConvNeXtBase(ConvNeXt):
    def __init__(self, hparams, **kwargs):
        hparams.block_setting = _block_setting(128, 256, 512, 1024)
        hparams.stochastic_depth_prob = 0.5
        super().__init__(hparams, **kwargs)


@model('convnext_large')
class ConvNeXtLarge(ConvNeXt):
    def __init__(self, hparams, **kwargs):
        hparams.block_setting = _block_setting(192, 384, 768, 1536)
        hparams.stochastic_depth_prob = 0.5
        super().__init__(hparams, **kwargs)
