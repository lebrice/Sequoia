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
def resize(x: Any, size: Tuple[int, ...], **kwargs) -> Any:
    """ Resizes a PIL.Image, a Tensor, ndarray, or a Box space. """
    raise NotImplementedError(f"Transform doesn't support input {x} of type {type(x)}")

@resize.register
def _(x: Image.Image, size: Tuple[int, ...], **kwargs) -> Image.Image:
    return F.resize(x, size, **kwargs)

@resize.register(np.ndarray)
@resize.register(Tensor)
def _(x: np.ndarray, size: Tuple[int, ...], **kwargs) -> np.ndarray:
    """ TODO: This resizes numpy arrays by converting them to tensors and then
    using the `interpolate` function. There is for sure a more efficient way to
    do this.
    """
    original = x
    if isinstance(original, np.ndarray):
        # Need to convert to tensor (for interpolate to work).
        x = torch.as_tensor(x)
    if len(original.shape) == 3:
        # Need to add a batch dimension (for interpolate to work).
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


@resize.register(tuple)
def _(x: Tuple[int, ...], size: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
    """ Give the resized image shape, given the input shape. """
    new_shape: Tuple[int, ...] = size
    if len(size) == 2:
        # Preserve the number of channels.
        if len(x) == 4:
            if has_channels_first(x):
                new_shape = [*x[:2], *size]
            elif has_channels_last(x):
                new_shape = [x[0], *size, x[-1]]
            else:
                raise NotImplementedError(x)
        elif len(x) == 3:
            if has_channels_first(x):
                new_shape = [x[0], *size]
            elif has_channels_last(x):
                new_shape = x[0]
                new_shape = [*size, x[-1]]
            else:
                raise NotImplementedError(x)
    else:
        NotImplementedError(size)
    return type(x)(new_shape)


@resize.register(spaces.Box)
def _(x: spaces.Box, size: Tuple[int, ...], **kwargs) -> spaces.Box:
    # Hmm, not sure if the bounds would actually also be respected though.
    return type(x)(
        low=resize(x.low, size, **kwargs),
        high=resize(x.high, size, **kwargs),
        dtype=x.dtype,
    )


class Resize(Resize_, Transform[Img, Img]):
    def __init__(self, size: Tuple[int, ...], interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)

    def __call__(self, img: Img) -> Img:
        return resize(img, size=self.size)