"""SpermYOLO backbone blocks.

The module path is intentionally retained for compatibility with checkpoints
saved by the original project.
"""

import torch.nn as nn

from ..modules.block import C3k, C3k2
from ..modules.conv import Conv
from .attention import EffectiveSEModule

__all__ = ("MBConv", "C3k_IDB", "C3k2_IDB")


class MBConv(nn.Module):
    def __init__(self, inc, ouc, shortcut=True, e=4, dropout=0.1):
        super().__init__()
        midc = inc * e
        self.conv_pw_1 = Conv(inc, midc, 1)
        self.conv_dw_1 = Conv(midc, midc, 3, g=midc)
        self.effective_se = EffectiveSEModule(midc)
        self.conv1 = Conv(midc, ouc, 1, act=False)
        self.dropout = nn.Dropout2d(p=dropout)
        self.add = shortcut and inc == ouc

    def forward(self, x):
        y = self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x)))))
        return x + y if self.add else y


class C3k_IDB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MBConv(c_, c_, shortcut) for _ in range(n)))


class C3k2_IDB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_IDB(self.c, self.c, 2, shortcut, g) if c3k else MBConv(self.c, self.c, shortcut)
            for _ in range(n)
        )
