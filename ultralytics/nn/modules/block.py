"""YOLOv11 building blocks required by the released detection YAMLs."""

import torch
import torch.nn as nn

from .conv import Conv


class DFL(nn.Module):
    """Distribution Focal Loss integral layer."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class SPPF(nn.Module):
    """Fast spatial pyramid pooling."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1, self.cv2 = Conv(c1, c_, 1, 1), Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard residual bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1, self.cv2 = Conv(c1, c_, k[0], 1), Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """Faster CSP bottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1, self.cv2 = Conv(c1, 2 * self.c, 1, 1), Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP bottleneck with three convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1, self.cv2, self.cv3 = Conv(c1, c_, 1, 1), Conv(c1, c_, 1, 1), Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3 bottleneck using configurable square kernels."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """YOLOv11 C2f block with optional C3k sub-blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class Attention(nn.Module):
    """Position-sensitive multi-head attention used by C2PSA."""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.key_dim, self.scale = int(self.head_dim * attn_ratio), int(self.head_dim * attn_ratio) ** -0.5
        nh_kd = self.key_dim * num_heads
        self.qkv, self.proj, self.pe = Conv(dim, dim + nh_kd * 2, 1, act=False), Conv(dim, dim, 1, act=False), Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).view(b, self.num_heads, self.key_dim * 2 + self.head_dim, h * w).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attention = (q.transpose(-2, -1) @ k * self.scale).softmax(dim=-1)
        x = (v @ attention.transpose(-2, -1)).view(b, c, h, w)
        return self.proj(x + self.pe(v.reshape(b, c, h, w)))


class PSABlock(nn.Module):
    """Position-sensitive attention and feed-forward block."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    """Stacked position-sensitive attention module."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1, self.cv2 = Conv(c1, 2 * self.c, 1, 1), Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), 1)
        return self.cv2(torch.cat((a, self.m(b)), 1))
