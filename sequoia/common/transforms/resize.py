from functools import singledispatch
from typing import Any, Callable, Sequence, Tuple, TypeVar, Union

import gym
import numpy as np
import torch
from gym import spaces
from PIL import Image
from sequoia.utils.logging_utils import get_logger
from torch import Tensor
from torch.nn.functional import interpolate
from torchvision.transforms import Resize as Resize_
from torchvision.transforms import ToTensor as ToTensor_
from torchvision.transforms import functional as F

from .channels import (channels_first, channels_last, has_channels_first,
                       has_channels_last)
from .transform import Img, Transform

logger = get_logger(__file__)

@singledispatch
def resize(x: Any, size: Tuple[int, ...]) -> Any:
    """ Resizes a PIL.Image, a Tensor, ndarray, or a Box space. """
    raise NotImplementedError(f"Transform doesn't support input {x} of type {type(x)}")

@resize.register
def _(x: Image.Image, size: Tuple[int, ...]) -> Image.Image:
    return F.resize(x, size)

@resize.register(np.ndarray)
@resize.register(Tensor)
def _(x: np.ndarray, size: Tuple[int, ...]) -> np.ndarray:
    """ TODO: This resizes numpy arrays by converting them to tensors and then
    using the `interpolate` function. There is for sure a more efficient way to
    do this.
    """
    original = x
    if isinstance(original, np.ndarray):
        x = torch.as_tensor(x)
    if len(original.shape) == 3:
        # Need to add a batch dimension.
        x = x.unsqueeze(0)
    if has_channels_last(original):
        # Need to make it channels first (for interpolate to work).
        x = channels_first(x)
                
    assert has_channels_first(x), f"Image needs to have channels first (shape is {x.shape})"
            
    x = interpolate(x, size, mode="area")
    if isinstance(original, np.ndarray):
        x = x.numpy()
    if len(original.shape) == 3:
        x = x[0]
    if has_channels_last(original):
        x = channels_last(x)
    return x


@resize.register(spaces.Box)
def _(x: spaces.Box, size: Tuple[int, ...]) -> spaces.Box:
    new_shape: Tuple[int, ...] = size
    if len(size) == 2:
        # Preserve the number of channels.
        if x.low.ndim == 4:
            if has_channels_first(x):
                new_shape = [*x.shape[:2], *size]
            elif has_channels_last(x):
                new_shape = [x.shape[0], *size, x.shape[-1]]
            else:
                raise NotImplementedError(x)
        elif x.low.ndim == 3:
            if has_channels_first(x):
                new_shape = [x.shape[0], *size]
            elif has_channels_last(x):
                new_shape = x.shape[0]
                new_shape = (*size, x.low.shape[-1])
            else:
                raise NotImplementedError(x)
    else:
        NotImplementedError(size)

    return type(x)(low=x.low.min(), high=x.high.max(), shape=new_shape, dtype=x.dtype)


class Resize(Resize_, Transform[Img, Img]):
    def __init__(self, size: Tuple[int, ...], interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)

    def __call__(self, img: Img) -> Img:
        return resize(img, size=self.size)
        # if not isinstance(img, Image.Image):
        #     original = img
        #     if isinstance(original, np.ndarray):
        #         img = torch.as_tensor(img)
        #     if len(original.shape) == 3:
        #         # Need to add a batch dimension.
        #         img = img.unsqueeze(0)
        #     if has_channels_last(original):
        #         # Need to make it channels first (for interpolate to work).
        #         img = ChannelsFirst.apply(img)
                
        #     assert has_channels_first(img), f"Image needs to have channels first (shape is {img.shape})"
            
        #     img = interpolate(img, self.size, mode="area")
        #     if isinstance(original, np.ndarray):
        #         img = img.numpy()
        #     if len(original.shape) == 3:
        #         img = img[0]
        #     if has_channels_last(original):
        #         img = ChannelsLast.apply(img)
        #     return img
        
        # return super().__call__(img)

    def shape_change(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert isinstance(self.size, (list, tuple)) and len(self.size) == 2,  "Can only tell the space change from resize when size has two values atm."
        if has_channels_first(input_shape):
            *n, c, h, w = input_shape
            return (*n, c, *self.size)
        if has_channels_last(input_shape):
            *n, h, w, c = input_shape
            return (*n, *self.size, c)
        raise NotImplementedError(f"Don't know what the shape change is for input shape {input_shape}")
