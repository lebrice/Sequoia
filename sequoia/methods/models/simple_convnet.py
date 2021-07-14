from sequoia.common.spaces.image import Image
from gym import spaces, Space
import torch
from torch import nn, Tensor
import numpy as np
from typing import List, Tuple
from sequoia.common.layers import Conv2d, Sequential


class SimpleConvNet(nn.Module):
    def __init__(self, in_channels: int=3, n_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.AdaptiveAvgPool2d(output_size=(8, 8)), # [16, 8, 8]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=False), # [32, 6, 6]
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False), # [32, 4, 4]
            nn.BatchNorm2d(32),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 120),  # NOTE: This '512' is what gets used as the
            # hidden size of the encoder.
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.features(x))
