"""SpermYOLO detection head.

Only detection is supported in the public minimal runtime.
"""

import math

import torch
import torch.nn as nn

from ..modules import Conv, DFL
from ..modules.conv import autopad
from ultralytics.utils.tal import dist2bbox, make_anchors
from .deconv import DEConv

__all__ = ("Scale", "Conv_GN", "DEConv_GN", "DESD_Head")


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class Conv_GN(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class DEConv_GN(DEConv):
    def __init__(self, dim):
        super().__init__(dim)
        self.bn = nn.GroupNorm(16, dim)


class DESD_Head(nn.Module):
    """Lightweight shared detail-enhanced convolutional detection head."""

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(c, hidc, 1)) for c in ch)
        self.share_conv = nn.Sequential(DEConv_GN(hidc), DEConv_GN(hidc))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.share_conv(self.conv[i](x[i]))
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:
            return x
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (xi.transpose(0, 1) for xi in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        y = torch.cat((self.decode_bboxes(box), cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
