"""OSNet-AIN, the person re-identification network `reid` embeds crops with.

Vendored from Torchreid (https://github.com/KaiyangZhou/deep-person-reid,
torchreid/models/osnet_ain.py) and cut to the one configuration used here, `osnet_ain_x1_0`,
for inference only: no classifier head and no weight download. Imported only by
`reid.load_model`, because torch is an optional extra.

Zhou et al., Omni-Scale Feature Learning for Person Re-Identification, ICCV 2019, and Learning
Generalisable Omni-Scale Representations for Person Re-Identification, TPAMI 2021.

MIT License

Copyright (c) 2018 Kaiyang Zhou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    """Convolution, then instance or batch norm, then ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        instance_norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn: nn.Module = (
            nn.InstanceNorm2d(out_channels, affine=True)
            if instance_norm
            else nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 convolution, batch norm, ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 convolution and, optionally, batch norm -- no non-linearity."""

    def __init__(self, in_channels: int, out_channels: int, bn: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x if self.bn is None else self.bn(x)


class LightConv3x3(nn.Module):
    """1x1 linear, then depthwise 3x3, batch norm, ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    """`depth` LightConv3x3 layers in a row."""

    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        layers = [LightConv3x3(in_channels, out_channels)]
        layers += [LightConv3x3(out_channels, out_channels) for _ in range(depth - 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ChannelGate(nn.Module):
    """Channel-wise sigmoid gates conditioned on the input."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))


class OSBlock(nn.Module):
    """Omni-scale block: four streams of growing receptive field, gated and summed.

    `instance_norm` is OSNet-AIN's variant (upstream `OSBlockINin`), which normalises inside
    the residual instead of batch-normalising the last projection.
    """

    def __init__(
        self, in_channels: int, out_channels: int, instance_norm: bool, reduction: int = 4
    ) -> None:
        super().__init__()
        mid = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, 5)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels, bn=not instance_norm)
        self.downsample = (
            Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        )
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if instance_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        gated = [self.gate(stream(x1)) for stream in self.conv2]
        x2 = gated[0]
        for g in gated[1:]:
            x2 = x2 + g
        x3 = self.conv3(x2)
        if self.IN is not None:
            x3 = self.IN(x3)
        identity = x if self.downsample is None else self.downsample(x)
        return F.relu(x3 + identity)


class OSNet(nn.Module):
    """OSNet-AIN x1.0 as a feature extractor: a 256 x 128 crop in, a 512-d embedding out.

    Attribute names follow upstream exactly, so its checkpoints load without renaming.
    """

    def __init__(self) -> None:
        super().__init__()
        c = (64, 256, 384, 512)
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3, instance_norm=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(OSBlock(c[0], c[1], True), OSBlock(c[1], c[1], True))
        self.pool2 = nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = nn.Sequential(OSBlock(c[1], c[2], False), OSBlock(c[2], c[2], True))
        self.pool3 = nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = nn.Sequential(OSBlock(c[2], c[3], True), OSBlock(c[3], c[3], False))
        self.conv5 = Conv1x1(c[3], c[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c[3], 512), nn.BatchNorm1d(512), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in (
            self.conv1,
            self.maxpool,
            self.conv2,
            self.pool2,
            self.conv3,
            self.pool3,
            self.conv4,
            self.conv5,
            self.global_avgpool,
        ):
            x = layer(x)
        return self.fc(x.flatten(1))
