from abc import ABC
from typing import Tuple

from torch import nn

from sequoia.common.layers import DeConvBlock, Reshape


class Decoder(nn.Sequential, ABC):
    """A base class for the decoders (mostly for typing purposes)."""

    code_size: int
    output_shape: Tuple[int, int, int]


class MnistDecoder(Decoder):
    """Decoder that generates images of shape [`out_channels`, 28, 28]"""

    def __init__(self, code_size: int, out_channels: int = 3):
        self.code_size = code_size
        self.output_shape: Tuple[int, int, int] = (out_channels, 28, 28)
        super().__init__(
            Reshape([self.code_size, 1, 1]),
            nn.ConvTranspose2d(self.code_size, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=1),
            nn.Sigmoid(),
        )


class CifarDecoder(Decoder):
    """Decoder that generates images of shape [3, 32, 32]"""

    def __init__(self, code_size: int):
        self.code_size = code_size
        self.output_shape: Tuple[int, int, int] = (3, 32, 32)
        super().__init__(
            Reshape([self.code_size, 1, 1]),
            DeConvBlock(self.code_size, 16),
            DeConvBlock(16, 32),
            DeConvBlock(32, 64),
            DeConvBlock(64, 64),
            DeConvBlock(64, 3, last_relu=False),
            nn.Sigmoid(),
        )


class ImageNetDecoder(Decoder):
    """Decoder that generates images of shape [3, 224, 224]"""

    def __init__(self, code_size: int):
        self.code_size = code_size
        self.output_shape: Tuple[int, int, int] = (3, 224, 224)
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
