"""Convolutional layers used by YOLOv11 and SpermYOLO."""

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Return padding that preserves spatial dimensions for stride-one convolution."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    return k // 2 if p is None and isinstance(k, int) else ([x // 2 for x in k] if p is None else p)


class Conv(nn.Module):
    """Convolution, batch normalization and activation."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=__import__("math").gcd(c1, c2), d=d, act=act)


class Concat(nn.Module):
    """Concatenate feature maps along a channel dimension."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
