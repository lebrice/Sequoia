# from torchvision.transforms import Lambda
from collections.abc import Mapping
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Callable, Iterable, Tuple, Union

import gym
import numpy as np
import torch
from gym import spaces
from sequoia.common.spaces import NamedTupleSpace
from sequoia.utils.logging_utils import get_logger
from torch import Tensor

from .transform import Img, Transform
from .utils import is_image

logger = get_logger(__file__)


def has_channels_last(img_or_shape: Union[Img, Tuple[int, ...], spaces.Box]) -> bool:
    """ Returns wether the given image, or image batch, shape, or Space is in
    the channels last format.
    """
    shape = getattr(img_or_shape, "shape", img_or_shape)
    return len(shape) and shape[-1] in {1, 3}


def has_channels_first(img_or_shape: Union[Img, Tuple[int, ...], spaces.Box]) -> bool:
    """ Returns wether the given image or image batch, shape, or Space is in
    the channels first format.
    """
    shape = getattr(img_or_shape, "shape", img_or_shape)
    if len(shape) == 3:
        return shape[0] in {1, 3}
    elif len(shape) == 4:
        return shape[1] in {1, 3}
    return False
    # return len(shape) and shape[0 if len(shape) == 3 else 1] in {1, 3}


def channels_last_if_needed(x: Any) -> Any:
    if has_channels_first(x):
        return channels_last(x)
    elif has_channels_last(x):
        return x
    raise RuntimeError(f"Input isn't channels_first or channels_last! {x.shape}")


def channels_first_if_needed(x: Any) -> Any:
    if has_channels_last(x):
        return channels_first(x)
    elif has_channels_first(x):
        return x
    raise RuntimeError(f"Input isn't channels_first or channels_last! {x.shape}")


class NamedDimensions(Transform[Tensor, Tensor]):
    """'Transform' that gives names to the dimensions of input tensors.
    Overwrites existing named dimensions, if any.
    """
    def __init__(self, names: Iterable[str]):
        self.names = tuple(names)
    
    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.refine_names(*self.names)

@singledispatch
def three_channels(x: Any) -> Any:
    """ Transform that makes the input images have three channels if they don't.
    
    * New: Also adds names to each dimension, when possible. (edit: off for now) 

    For instance, if the input shape is:
    [28, 28] -> [3, 28, 28] (copy the image three times)
    [1, 28, 28] -> [3, 28, 28] (same idea)
    [10, 1, 28, 28] -> [10, 3, 28, 28] (keep batch intact, do the same again.)
    
    """
    raise NotImplementedError(f"This doesn't currently support input {x} of type {type(x)}")

@three_channels.register(Tensor)
def _(x: Tensor) -> Tensor:
    names: Tuple[str, ...] = ()
    if x.ndim == 2:
        x = x.reshape([1, *x.shape])
        x = x.repeat(3, 1, 1)
        names = ("C", "H", "W")
    if x.ndim == 3:
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
            names = ("C", "H", "W")
        elif x.shape[-1] == 1:
            x = x.repeat(1, 1, 3)
            names = ("H", "W", "C")
    if x.ndim == 4:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            names = ("N", "C", "H", "W")
        elif x.shape[-1] == 1:
            x = x.repeat(1, 1, 1, 3)
            names = ("N", "H", "W", "C")
    # FIXME: Turning this off for now, since using named dimensions
    # generates a whole lot of UserWarnings atm.
    # if isinstance(x, Tensor) and names:
    #     # Cool new pytorch feature!
    #     x.rename(*names)
    return x

@three_channels.register(np.ndarray)
def _(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        # names = ("C", "H", "W")
        x = x.reshape([1, *x.shape])
        x = np.tile(x, [3, 1, 1])
    if x.ndim == 3:
        if x.shape[0] == 1:
            # names = ("C", "H", "W")
            x = np.tile(x, [3, 1, 1])
        elif x.shape[-1] == 1:
            # names = ("H", "W", "C")
            x = np.tile(x, [1, 1, 3])
    if x.ndim == 4:
        if x.shape[1] == 1:
            # names = ("N", "C", "H", "W")
            x = np.tile(x, [1, 3, 1, 1])
        elif x.shape[-1] == 1:
            # names = ("N", "H", "W", "C")
            x = np.tile(x, [1, 1, 1, 3])
    return x

@three_channels.register(spaces.Box)
def _(x: spaces.Box) -> spaces.Box:
    return type(x)(low=three_channels(x.low), high=three_channels(x.high), dtype=x.dtype)


@three_channels.register(torch.Size)
@three_channels.register(tuple)
def _(x: Tuple[int, ...]) -> Tuple[int, ...]:
    dims = len(x)
    if dims == 2:
        return (3, *x)
    elif dims == 3:
        if x[0] == 1:
            return (3, *x[1:])
        elif x[-1] == 1:
            return (*x[:-1], 3)
    elif dims == 4:
        if x[1] == 1:
            return (x[0], 3, *x[2:])
        elif x[-1] == 1:
            return (*x[:-1], 3)
    return x



@three_channels.register(NamedTupleSpace)
@three_channels.register(spaces.Dict)
@three_channels.register(Mapping)
def _three_channels(x: Any) -> Any:
    return type(x)(**{
        key: three_channels(value) if is_image(value) else value
        for key, value in x.items()
    })


@dataclass
class ThreeChannels(Transform[Tensor, Tensor]):
    """ Transform that makes the input images have three tensors.
    
    * New: Also adds names to each dimension, when possible.
    
    For instance, if the input shape is:
    [28, 28] -> [3, 28, 28] (copy the image three times)
    [1, 28, 28] -> [3, 28, 28] (same idea)
    [10, 1, 28, 28] -> [10, 3, 28, 28] (keep batch intact, do the same again.)
    
    """
    def __call__(self, x: Tensor) -> Tensor:
        return three_channels(x)
        



@singledispatch
def channels_first(x: Any) -> Any:
    """ Re-orders the dimensions of the input from ((n), H, W, C) to ((n), C, H, W).
    If the tensor doesn't have named dimensions, this will ALWAYS re-order the
    dimensions, regarless of if the image or space already has channels first.

    Also converts non-Tensor inputs to tensors using `to_tensor`.
    """
    raise RuntimeError(f"Transform isn't applicable to input {x} of type {type(x)}.")

@channels_first.register(Tensor)
def _(x: Tensor) -> Tensor:
    if x.ndim == 3:
        if any(x.names):
            return x.align_to("C", "H", "W")
        return x.permute(2, 0, 1)#.to(memory_format=torch.contiguous_format)
    if x.ndim == 4:
        if any(x.names):
            return x.align_to("N", "C", "H", "W")
        return x.permute(0, 3, 1, 2).contiguous()
    return x

@channels_first.register(tuple)
def _(x: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(x) == 3:
        # TODO: Re-enable the naming of the dimensions at some point.
        return type(x)(x[i] for i in (2, 0, 1))
    if len(x.shape) == 4:
        return type(x)(x[i] for i in (0, 3, 1, 2))
    raise NotImplementedError(x)


@channels_first.register(np.ndarray)
def _(x: spaces.Box) -> spaces.Box:
    if x.ndim == 4:
        return np.moveaxis(x, 3, 1)  
    elif x.ndim == 3:
        return np.moveaxis(x, 2, 0)
    else:
        raise NotImplementedError(f"Expected 3-d or 4-d input, got {x}")


@channels_first.register(tuple)
def _(x: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(x) == 4:
        return type(x)(x[i] for i in (0, 3, 1, 2))
    if len(x) == 3:
        return type(x)(x[i] for i in (2, 0, 1))
    raise NotImplementedError(x)


@channels_first.register(spaces.Box)
def _(x: spaces.Box) -> spaces.Box:
    return type(x)(
        low=channels_first(x.low),
        high=channels_first(x.high),
        dtype=x.dtype,
    )


@dataclass
class ChannelsFirst(Transform[Union[np.ndarray, Tensor], Tensor]):
    """ Re-orders the dimensions of the tensor from ((n), H, W, C) to ((n), C, H, W).
    If the tensor doesn't have named dimensions, this will ALWAYS re-order the
    dimensions, regarless of the length of the last dimension.

    Also converts non-Tensor inputs to tensors using `to_tensor`.
    """
    def __call__(self, x: Tensor) -> Tensor:
        return self.apply(x)
        
    @classmethod
    def apply(cls, x: Tensor) -> Tensor:
        return channels_first(x)

        # if not isinstance(x, Tensor):
        #     raise RuntimeError(f"Transform only applies to Tensors. (Not {x} of type {type(x)}).")
        
        # # if has_channels_first(x):
        # #     logger.warning(RuntimeWarning(f"Input already seems to have channels first, but this transform will be applied anyway.."))

        # if x.ndim == 3:
        #     if any(x.names):
        #         return x.align_to("C", "H", "W")
        #     return x.permute(2, 0, 1)#.to(memory_format=torch.contiguous_format)
        # if x.ndim == 4:
        #     if any(x.names):
        #         return x.align_to("N", "C", "H", "W")
        #     return x.permute(0, 3, 1, 2).contiguous()
        # return x
    
    # @staticmethod
    # def shape_change(input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
    #     ndim = len(input_shape)
    #     if ndim == 3:
    #         return tuple(input_shape[i] for i in (2, 0, 1))
    #     elif ndim == 4:
    #         return tuple(input_shape[i] for i in (0, 3, 1, 2))         
    #     return input_shape

@dataclass
class ChannelsFirstIfNeeded(ChannelsFirst):
    """ Only puts the channels first if the input has channels last. """
    
    @classmethod
    def apply(cls, x: Tensor) -> Tensor:
        if has_channels_last(x):
            return super().apply(x)
        return x

    # @classmethod
    # def shape_change(cls, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
    #     if has_channels_last(input_shape):
    #         return super().shape_change(input_shape)
    #     return input_shape

@singledispatch
def channels_last(x: Any) -> Any:
    raise NotImplementedError(f"This doesn't support input {x} of type {type(x)}")


@channels_last.register(Tensor)
def _(x: Tensor) -> Tensor:
    if len(x.shape) == 3:
        # TODO: Re-enable the naming of the dimensions at some point.
        # if not x.names:
        #     x.rename("C", "H", "W")
        #     return x.align_to("H", "W", "C")
        return x.permute(1, 2, 0)
    if len(x.shape) == 4:
        return x.permute(0, 2, 3, 1)


@channels_last.register(tuple)
def _(x: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(x) == 3:
        # TODO: Re-enable the naming of the dimensions at some point.
        return type(x)(x[i] for i in (1, 2, 0))
    if len(x.shape) == 4:
        return type(x)(x[i] for i in (0, 2, 3, 1))
    raise NotImplementedError(x)


@channels_last.register(np.ndarray)
def _(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 4:
        return np.moveaxis(x, 1, 3)
    elif len(x.shape) == 3:
        return np.moveaxis(x, 0, 2)
    raise NotImplementedError(x.shape)


@channels_last.register(spaces.Box)
def _(x: spaces.Box) -> spaces.Box:
    return type(x)(
        low=channels_last(x.low),
        high=channels_last(x.high),
        dtype=x.dtype,
    )


@dataclass
class ChannelsLast(Transform[Tensor, Tensor]):
    def __call__(self, x: Tensor) -> Tensor:
        return self.apply(x)

    @classmethod
    def apply(cls, x: Tensor) -> Tensor:
        return channels_last(x)

@dataclass
class ChannelsLastIfNeeded(ChannelsLast):
    """ Only puts the channels last if the input has channels first. """
    @classmethod
    def apply(cls, x: Tensor) -> Tensor:
        return channels_last_if_needed(x)
