from typing import Dict, Type, Union, Tuple

import torch
from torch import nn

from .decoders import CifarDecoder, ImageNetDecoder, MnistDecoder
from common.dims import Dims

import functools
def get_decoder_class_for_dataset(input_shape: Union[Tuple[int, int, int], Dims]) -> Type[nn.Module]:
    assert len(input_shape) == 3, input_shape
    if not isinstance(input_shape, Dims):
        assert input_shape[0] == min(input_shape), f"should be in C, H, W format: {input_shape}"
        input_shape = Dims(c=input_shape[0], h=input_shape[1], w=input_shape[2])  # type: ignore
    if input_shape.h == 28:
        return functools.partial(MnistDecoder, out_channels=input_shape.c)
    elif input_shape.h == 32:
        return CifarDecoder
    elif input_shape.h == 224:
        return ImageNetDecoder
    else:
        raise RuntimeError(f"No decoder available for input shape {input_shape}")
