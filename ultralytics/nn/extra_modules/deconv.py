"""Detail-enhanced convolution used by the SpermYOLO detection head."""

import torch
from torch import nn

from ..modules.conv import Conv


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.theta = theta

    def get_weight(self):
        weight = self.conv.weight
        out_channels, in_channels, height, width = weight.shape
        flattened = weight.reshape(out_channels, in_channels, -1)
        result = flattened.clone()
        result[:, :, 4] = flattened[:, :, 4] - flattened.sum(dim=2)
        return result.reshape(out_channels, in_channels, height, width), self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.theta = theta

    def get_weight(self):
        weight = self.conv.weight
        out_channels, in_channels, height, width = weight.shape
        flattened = weight.reshape(out_channels, in_channels, -1)
        result = flattened - self.theta * flattened[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        return result.reshape(out_channels, in_channels, height, width), self.conv.bias


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        weight = self.conv.weight
        out_channels, in_channels, kernel = weight.shape
        result = weight.new_zeros(out_channels, in_channels, kernel * kernel)
        result[:, :, [0, 3, 6]] = weight
        result[:, :, [2, 5, 8]] = -weight
        return result.reshape(out_channels, in_channels, kernel, kernel), self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        weight = self.conv.weight
        out_channels, in_channels, kernel = weight.shape
        result = weight.new_zeros(out_channels, in_channels, kernel * kernel)
        result[:, :, [0, 1, 2]] = weight
        result[:, :, [6, 7, 8]] = -weight
        return result.reshape(out_channels, in_channels, kernel, kernel), self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(dim)
        self.act = Conv.default_act

    def forward(self, x):
        if hasattr(self, "conv1_1"):
            weights_biases = [module.get_weight() for module in (self.conv1_1, self.conv1_2, self.conv1_3, self.conv1_4)]
            weights_biases.append((self.conv1_5.weight, self.conv1_5.bias))
            weight = sum(item[0] for item in weights_biases)
            bias = sum(item[1] for item in weights_biases)
            result = nn.functional.conv2d(x, weight, bias, stride=1, padding=1)
        else:
            result = self.conv1_5(x)
        return self.act(self.bn(result)) if hasattr(self, "bn") else self.act(result)

    def switch_to_deploy(self):
        weights_biases = [module.get_weight() for module in (self.conv1_1, self.conv1_2, self.conv1_3, self.conv1_4)]
        weights_biases.append((self.conv1_5.weight, self.conv1_5.bias))
        self.conv1_5.weight = nn.Parameter(sum(item[0] for item in weights_biases))
        self.conv1_5.bias = nn.Parameter(sum(item[1] for item in weights_biases))
        del self.conv1_1, self.conv1_2, self.conv1_3, self.conv1_4
