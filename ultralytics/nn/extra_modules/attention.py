"""Attention primitives required by the released SpermYOLO architecture."""

import torch.nn as nn

__all__ = ("EffectiveSEModule",)


class EffectiveSEModule(nn.Module):
    """Efficient squeeze-excitation used inside the EMBC block."""

    def __init__(self, channels, add_maxpool=False):
        super().__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        return x * self.gate(self.fc(x_se))
