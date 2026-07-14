"""Spectral feed-forward layer retained under its legacy checkpoint path."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("SpectralEnhancedFFN",)


class SpectralEnhancedFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=2,
            groups=hidden_features * 2,
            bias=bias,
            dilation=2,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))
        self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))

    @staticmethod
    def _pad_width(x, factor):
        width = x.shape[-1]
        padding = 0 if width % factor == 0 else (width // factor + 1) * factor - width
        return F.pad(x, (0, padding), "constant", 0), padding

    def forward(self, x):
        dtype = x.dtype
        x = self.dwconv(self.project_in(x))
        x, padding = self._pad_width(x, 2)
        x = torch.fft.rfft2(x.float())
        x = self.fft_channel_weight * x + self.fft_channel_bias
        x = torch.fft.irfft2(x)
        if padding:
            x = x[..., :-padding]
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out((F.silu(x1) * x2).to(dtype))
