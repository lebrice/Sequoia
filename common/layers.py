from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn


class Lambda(nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view([inputs.shape[0], -1])


class Reshape(nn.Module):
    def __init__(self, target_shape: Union[List[int], Tuple[int, ...]]):
        self.target_shape = target_shape
        super().__init__()

    def forward(self, inputs):
        return inputs.reshape([inputs.shape[0], *self.target_shape])


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int=3,
                 padding: int=1,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return self.pool(x)

class DeConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: Optional[int]=None,
                 kernel_size: int=3,
                 padding: int=1,
                 **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels or out_channels
        self.kernel_size = kernel_size
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs,
        )
        self.norm1 = nn.BatchNorm2d(self.hidden_channels)
        self.conv2 = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs,
        )
        self.norm2 = nn.BatchNorm2d(self.hidden_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x
