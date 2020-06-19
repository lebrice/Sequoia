from typing import Tuple, Union

import torch
from torch import nn

from common.layers import DeConvBlock, Reshape
from tasks.auxiliary_task import AuxiliaryTask
from datasets import Datasets
from abc import ABC, abstractmethod


class MnistDecoder(nn.Sequential):
    def __init__(self, code_size: int):
        self.code_size = code_size
        super().__init__(
            Reshape([self.code_size, 1, 1]),
            nn.ConvTranspose2d(self.code_size, 32, kernel_size=4 , stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(16,16,kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=1),
            nn.Sigmoid(),
        )


class CifarDecoder(nn.Sequential):
    def __init__(self, code_size: int):
        self.code_size = code_size
        super().__init__(
            Reshape([self.code_size, 1, 1]),
            DeConvBlock(self.code_size, 16),
            DeConvBlock(16, 32),
            DeConvBlock(32, 64),
            DeConvBlock(64, 64),
            DeConvBlock(64, 3, last_relu=False),
            nn.Sigmoid(),
        )


class ImageNetDecoder(nn.Sequential):
    def __init__(self, code_size: int):
        self.code_size = code_size
        super().__init__(
            Reshape([self.code_size, 1, 1]),
            DeConvBlock(self.code_size, 16),
            DeConvBlock(16, 32),
            DeConvBlock(32, 64),
            DeConvBlock(64, 128),
            DeConvBlock(128, 224),
            DeConvBlock(224, 3, last_relu=False),
            nn.Sigmoid(),
        )

