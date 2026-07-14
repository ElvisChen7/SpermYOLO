"""Standard YOLO detection head."""

import math

import torch
import torch.nn as nn

from ultralytics.utils.tal import dist2bbox, make_anchors

from .block import DFL
from .conv import Conv, DWConv


class Detect(nn.Module):
    """Anchor-free YOLO detection head."""

    dynamic = False
    export = False
    end2end = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc, self.nl, self.reg_max = nc, len(ch), 16
        self.no, self.stride = nc + self.reg_max * 4, torch.zeros(self.nl)
        c2, c3 = max(16, ch[0] // 4, self.reg_max * 4), max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1), DWConv(c3, c3, 3), Conv(c3, c3, 1), nn.Conv2d(c3, nc, 1))
            for x in ch
        )
        self.dfl = DFL(self.reg_max)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        return (self._inference(x), x)

    def _inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (tensor.transpose(0, 1) for tensor in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box[-1].bias.data[:] = 1.0
            cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)

    def decode_bboxes(self, bboxes, anchors):
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)
