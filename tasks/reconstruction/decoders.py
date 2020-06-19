from typing import Tuple, Union

import torch
from torch import nn

from common.layers import DeConvBlock, Reshape
from tasks.auxiliary_task import AuxiliaryTask
from datasets import Datasets

def get_decoder(input_size: Union[Tuple[int, int, int], torch.Size], code_size: int) -> nn.Module:
    if input_size == Datasets.mnist.value.x_shape:
        # TODO: get the right decoder architecture for other datasets than MNIST.
        return MnistDecoder(code_size=code_size)
    elif input_size == Datasets.cifar10.value.x_shape:
        return CifarDecoder(code_size=code_size)
    else:
        raise RuntimeError(f"Don't have a decoder for the given input shape: {input_size}")


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
