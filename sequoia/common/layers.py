import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from sequoia.common.spaces.image import Image
from sequoia.common.transforms import Compose
from sequoia.utils.generic_functions import singledispatchmethod
from sequoia.utils.logging_utils import get_logger
from torch import Tensor, nn
from torch.nn import Flatten

logger = get_logger(__file__)
import torch
from torch import Tensor, nn


class Lambda(nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)


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
    """Block that performs:
    Upsample (2x)
    Conv
    BatchNorm2D
    Relu
    Conv
    BatchNorm2D
    Relu (optional)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: Optional[int]=None,
                 kernel_size: int=3,
                 padding: int=1,
                 last_relu: bool=True,
                 **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels or out_channels
        self.kernel_size = kernel_size
        self.last_relu = last_relu
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
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
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.last_relu:
            x = self.relu(x)
        return x




def n_output_features(
    in_features: int, padding: int = 1, kernel_size: int = 3, stride: int = 1
) -> int:
    """ Calculates the number of output features of a conv2d layer given its parameters.
    """
    return math.floor((in_features + 2 * padding - kernel_size) / stride) + 1




class Conv2d(nn.Conv2d):
    @singledispatchmethod
    def forward(self, input: Union[Image, Tensor]) -> Union[Tensor, Image]:
        return super().forward(input)

    @forward.register(Image)
    def _(self, input: Image) -> Image:
        assert input.channels_first, f"Need channels first inputs for conv2d: {input}"
        # NOTE: Not strictly necessary for computing the output space, but it would be
        # better for the input space to already have a batch size, since conv2d only
        # accepts 4-dimensional inputs.
        # assert input.batch_size, (
        #     f"Image space should be batched, since conv2d only accepts 4-dimensional "
        #     f"inputs. (input={input})"
        # )
        assert input.channels == self.in_channels, (
            f"Input space doesn't have the right number of channels: "
            f"input.channels: {input.channels} != self.in_channels: {self.in_channels}"
        )
        new_height = n_output_features(
            input.height,
            padding=self.padding[0],
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
        )
        new_width = n_output_features(
            input.width,
            padding=self.padding[1],
            kernel_size=self.kernel_size[1],
            stride=self.stride[1],
        )
        new_channels = self.out_channels

        new_shape = [new_channels, new_height, new_width]
        if input.batch_size:
            new_shape.insert(0, input.batch_size)

        output_space: Image = type(input)(low=-np.inf, high=np.inf, shape=new_shape)
        output_space.channels_first = True
        return output_space


class MaxPool2d(nn.MaxPool2d):
    @singledispatchmethod
    def forward(self, input: Union[Image, Tensor]) -> Union[Tensor, Image]:
        return super().forward(input)

    @forward.register(Image)
    def _(self, input: Image) -> Image:
        assert input.channels_first, f"Need channels first inputs: {input}"
        # assert not self.padding, "assuming no padding for now."
        padding = [self.padding] * 2 if isinstance(self.padding, int) else self.padding 
        kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size 
        stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride 
        
        new_height = n_output_features(
            input.height,
            padding=padding[0],
            kernel_size=kernel_size[0],
            stride=stride[0],
        )
        new_width = n_output_features(
            input.width,
            padding=padding[1],
            kernel_size=kernel_size[1],
            stride=stride[1],
        )

        new_channels = input.channels
        new_shape = [new_channels, new_height, new_width]
        if input.batch_size:
            new_shape.insert(0, input.batch_size)
        output_space: Image = type(input)(low=-np.inf, high=np.inf, shape=new_shape)
        output_space.channels_first = True
        # assert False, (self.forward(torch.as_tensor([input.sample()])).shape, output_space)
        return output_space


class Sequential(nn.Sequential):
    
    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, input):
        if isinstance(input, spaces.Space):
            space = input
            for module in self:
                try:
                    space = module(space)
                except:
                    if isinstance(space, (spaces.Box, Image)):
                        # Apply the module to a sample from the space, and create an
                        # output space of the same shape.
                        space = Image.from_box(space)
                        in_sample: Tensor = torch.as_tensor(space.sample())
                        if not space.batch_size:
                            in_sample = in_sample.unsqueeze(0)
                        out_sample = module(in_sample)
                        out_space = type(space)(low=-np.inf, high=np.inf, shape=out_sample.shape)
                        space = out_space
                    else:
                        logger.debug(f"Unable to apply module {module} on space {space}: assuming that it doesn't change the space.")
            return space
        return super().forward(input)

