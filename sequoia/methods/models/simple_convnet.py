from sequoia.common.spaces.image import Image
from gym import spaces, Space
import torch
from torch import nn, Tensor
import numpy as np
from typing import List, Tuple
from sequoia.common.layers import Conv2d, Sequential

def get_convnet(
    image_space: Image, desired_representation_size: spaces.Box
) -> Tuple[nn.Module, spaces.Box]:
    """ WIP: Idea: create a convnet dynamically, depending on the input space and the
    desired output space.
    """
    conv_layers: List[nn.Module] = []
    image_space = Image.from_box(image_space)

    w = image_space.w
    h = image_space.h
    c = image_space.c

    r = desired_representation_size
        
    layers = []
    
    space = image_space
    for i in range(3):
        input_dims = space.c * space.w * space.h
        out_c = max(8, min(space.c * 2, 256))

        kernel_size = [3, 3]
        stride = [1, 1]
        padding = [1, 1]

        if space.h > 32:
            kernel_size[0] = 5
            stride[0] = 2
        if space.w > 32:
            kernel_size[1] = 5
            stride[1] = 2
        import torchvision
        assert False, torchvision.models.resnet18()
        conv = Conv2d(space.c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        block = Sequential(
            conv,
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        # NOTE: Since we use the 'Compose' class, we can apply the list to an input.
        space = block(space)
        layers.append(block)

    repr_space = space
        

    assert False, repr_space


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
