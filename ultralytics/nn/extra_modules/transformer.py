"""SpermYOLO PSA block with spectral feed-forward layers."""

import torch.nn as nn

from ..modules.block import C2PSA, PSABlock
from .transMamba import SpectralEnhancedFFN

__all__ = ("D2SEMBlock", "D2SEM")


class D2SEMBlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.ffn = SpectralEnhancedFFN(c, 2, False)


class D2SEM(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(D2SEMBlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
