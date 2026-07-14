"""Multi-feature fusion module required by SpermYOLO."""

import torch
import torch.nn as nn

from ..modules.conv import Conv

__all__ = ("MFM",)


class MFM(nn.Module):
    """Attention-weighted multi-feature fusion."""

    def __init__(self, inc, dim, reduction=8):
        super().__init__()
        self.height = len(inc)
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * self.height, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)
        self.conv1x1 = nn.ModuleList(Conv(c, dim, 1) if c != dim else nn.Identity() for c in inc)

    def forward(self, in_feats):
        feats = [layer(feature) for layer, feature in zip(self.conv1x1, in_feats)]
        batch, channels, height, width = feats[0].shape
        feats = torch.cat(feats, dim=1).view(batch, self.height, channels, height, width)
        attention = self.mlp(self.avg_pool(feats.sum(dim=1)))
        attention = self.softmax(attention.view(batch, self.height, channels, 1, 1))
        return (feats * attention).sum(dim=1)
