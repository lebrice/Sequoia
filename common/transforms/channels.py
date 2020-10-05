from dataclasses import dataclass
from typing import Callable, Union, Callable, Tuple
from torch import Tensor
import numpy as np
import torch

@dataclass
class ThreeChannels(Callable[[Tensor], Tensor]):
    """ Transform that makes the input images have three tensors.
    
    For instance, if the input shape is:
    [28, 28] -> [3, 28, 28] (copy the image three times)
    [1, 28, 28] -> [3, 28, 28] (same idea)
    [10, 1, 28, 28] -> [10, 3, 28, 28] (keep batch intact, do the same again.)
    
    """
    def __call__(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.reshape([1, *x.shape])
            x = x.repeat(3, 1, 1)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x
    
    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        dims = len(input_shape)
        if dims == 2:
            return (3, *input_shape)
        elif dims == 3 and input_shape[0] == 1:
            return (3, *input_shape[1:])
        elif dims == 4 and input_shape[1] == 1:
            return (input_shape[0], 3, *input_shape[2:])
        return input_shape


@dataclass
class ChannelsFirst(Callable[[Union[np.ndarray, Tensor]], Tensor]):
    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = to_tensor(x)
        if x.ndim == 3:
            if not x.names:
                x.rename("H", "W", "C")
                return x.align_to("C", "H", "W")
            return x.permute(2, 0, 1)#.to(memory_format=torch.contiguous_format)
        if x.ndim == 4:
            return x.permute(0, 3, 1, 2).contiguous()
        return x
    
    @staticmethod
    def shape_change(input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        ndim = len(input_shape)
        if ndim == 3:
            return tuple(input_shape[i] for i in (2, 0, 1))
        elif ndim == 4:
            return tuple(input_shape[i] for i in (0, 3, 1, 2))         
        return input_shape

@dataclass
class ChannelsFirstIfNeeded(ChannelsFirst):
    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = to_tensor(x)
        if x.shape[-1] in {1, 3}:
            return super().__call__(x)
        return x

    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        if input_shape[-1] in {1, 3}:
            return super().shape_change(input_shape)
        else:
            return input_shape


@dataclass
class ChannelsLast(Callable[[Tensor], Tensor]):
    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = to_tensor(x)
        if x.ndimension() == 3:
            if not x.names:
                x.rename("C", "H", "W")
                return x.align_to("H", "W", "C")
            return x.permute(1, 2, 0)
        if x.ndimension() == 4:
            return x.permute(0, 2, 3, 1)
        return x
    
    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        ndim = len(input_shape)
        if ndim == 3:
            new_shape = tuple(input_shape[i] for i in (1, 2, 0))
        elif ndim == 4:
            new_shape = tuple(input_shape[i] for i in (0, 2, 3, 1))
        return new_shape

@dataclass
class ChannelsLastIfNeeded(ChannelsLast):
    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = to_tensor(x)
        if x.ndimension() == 4 and x.shape[1] in {1, 3}:
            return super().__call__(x)
        if x.ndimension() == 3 and x.shape[0] in {1, 3}:
            return super().__call__(x)
        return x
    
    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        ndims = len(input_shape)
        if ndims == 4 and input_shape[1] in {1, 3}:
            return super().shape_change(input_shape)
        if ndims == 3 and input_shape[0] in {1, 3}:
            return super().shape_change(input_shape)
        return input_shape

